use std::cmp::Ordering;

#[derive(Debug, Clone, serde::Serialize)]
pub struct AlignedToken {
    pub id: usize,
    pub text: String,
    pub start: f64,
    pub duration: f64,
    pub confidence: f64,
    pub end: f64,
}

impl AlignedToken {
    pub fn new(id: usize, text: String, start: f64, duration: f64, confidence: f64) -> Self {
        let end = start + duration;
        Self {
            id,
            text,
            start,
            duration,
            confidence,
            end,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AlignedSentence {
    pub text: String,
    pub tokens: Vec<AlignedToken>,
    pub start: f64,
    pub end: f64,
    pub duration: f64,
    pub confidence: f64,
}

impl AlignedSentence {
    pub fn new(text: String, mut tokens: Vec<AlignedToken>) -> Self {
        tokens.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(Ordering::Equal));
        let start = tokens.first().map(|t| t.start).unwrap_or(0.0);
        let end = tokens.last().map(|t| t.end).unwrap_or(0.0);
        let duration = end - start;
        let confidence = if tokens.is_empty() {
            1.0
        } else {
            let log_sum = tokens
                .iter()
                .map(|t| (t.confidence + 1e-10).ln())
                .sum::<f64>();
            (log_sum / tokens.len() as f64).exp()
        };
        Self {
            text,
            tokens,
            start,
            end,
            duration,
            confidence,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AlignedResult {
    pub text: String,
    pub sentences: Vec<AlignedSentence>,
}

impl AlignedResult {
    pub fn new(text: String, sentences: Vec<AlignedSentence>) -> Self {
        Self {
            text: text.trim().to_string(),
            sentences,
        }
    }

    pub fn tokens(&self) -> Vec<AlignedToken> {
        self.sentences
            .iter()
            .flat_map(|s| s.tokens.clone())
            .collect()
    }

    /// Iterate over all tokens without cloning. Prefer this over `tokens()`
    /// when ownership is not needed.
    pub fn iter_tokens(&self) -> impl Iterator<Item = &AlignedToken> {
        self.sentences.iter().flat_map(|s| s.tokens.iter())
    }
}

#[derive(Debug, Clone, Default)]
pub struct SentenceConfig {
    pub max_words: Option<usize>,
    pub silence_gap: Option<f64>,
    pub max_duration: Option<f64>,
}

#[must_use]
pub fn tokens_to_sentences(
    tokens: &[AlignedToken],
    config: &SentenceConfig,
) -> Vec<AlignedSentence> {
    let mut sentences = Vec::new();
    let mut current_tokens: Vec<AlignedToken> = Vec::new();

    for (idx, token) in tokens.iter().enumerate() {
        current_tokens.push(token.clone());

        let is_punctuation = token.text.contains('!')
            || token.text.contains('?')
            || token.text.contains('。')
            || token.text.contains('？')
            || token.text.contains('！')
            || (token.text.contains('.')
                && (idx == tokens.len() - 1
                    || tokens
                        .get(idx + 1)
                        .map(|t| t.text.contains(' '))
                        .unwrap_or(false)));

        let is_word_limit = if let Some(max_words) = config.max_words {
            if idx != tokens.len() - 1 {
                let words_in_current = current_tokens
                    .iter()
                    .filter(|t| t.text.contains(' '))
                    .count();
                let next_is_word = tokens
                    .get(idx + 1)
                    .map(|t| t.text.contains(' '))
                    .unwrap_or(false);
                words_in_current + if next_is_word { 1 } else { 0 } > max_words
            } else {
                false
            }
        } else {
            false
        };

        let is_long_silence = if let Some(gap) = config.silence_gap {
            if idx != tokens.len() - 1 {
                tokens[idx + 1].start - token.end >= gap
            } else {
                false
            }
        } else {
            false
        };

        let is_over_duration = if let Some(max_dur) = config.max_duration {
            let start_time = current_tokens
                .first()
                .map(|t| t.start)
                .unwrap_or(token.start);
            token.end - start_time >= max_dur
        } else {
            false
        };

        if is_punctuation || is_word_limit || is_long_silence || is_over_duration {
            let sentence_text = current_tokens.iter().map(|t| t.text.clone()).collect();
            sentences.push(AlignedSentence::new(sentence_text, current_tokens));
            current_tokens = Vec::new();
        }
    }

    if !current_tokens.is_empty() {
        let sentence_text = current_tokens.iter().map(|t| t.text.clone()).collect();
        sentences.push(AlignedSentence::new(sentence_text, current_tokens));
    }

    sentences
}

#[must_use]
pub fn sentences_to_result(sentences: &[AlignedSentence]) -> AlignedResult {
    let text = sentences.iter().map(|s| s.text.clone()).collect::<String>();
    AlignedResult::new(text, sentences.to_vec())
}

pub fn merge_longest_contiguous(
    a: &[AlignedToken],
    b: &[AlignedToken],
    overlap_duration: f64,
) -> Result<Vec<AlignedToken>, String> {
    if a.is_empty() {
        return Ok(b.to_vec());
    }
    if b.is_empty() {
        return Ok(a.to_vec());
    }

    let a_end_time = a.last().unwrap().end;
    let b_start_time = b.first().unwrap().start;

    if a_end_time <= b_start_time {
        let mut out = a.to_vec();
        out.extend_from_slice(b);
        return Ok(out);
    }

    let overlap_a: Vec<_> = a
        .iter()
        .filter(|t| t.end > b_start_time - overlap_duration)
        .cloned()
        .collect();
    let overlap_b: Vec<_> = b
        .iter()
        .filter(|t| t.start < a_end_time + overlap_duration)
        .cloned()
        .collect();

    let enough_pairs = overlap_a.len() / 2;

    if overlap_a.len() < 2 || overlap_b.len() < 2 {
        let cutoff_time = (a_end_time + b_start_time) / 2.0;
        let mut out: Vec<AlignedToken> =
            a.iter().filter(|t| t.end <= cutoff_time).cloned().collect();
        out.extend(b.iter().filter(|t| t.start >= cutoff_time).cloned());
        return Ok(out);
    }

    let mut best_contiguous: Vec<(usize, usize)> = Vec::new();
    for i in 0..overlap_a.len() {
        for j in 0..overlap_b.len() {
            if overlap_a[i].id == overlap_b[j].id
                && (overlap_a[i].start - overlap_b[j].start).abs() < overlap_duration / 2.0
            {
                let mut current: Vec<(usize, usize)> = Vec::new();
                let mut k = i;
                let mut l = j;
                while k < overlap_a.len()
                    && l < overlap_b.len()
                    && overlap_a[k].id == overlap_b[l].id
                    && (overlap_a[k].start - overlap_b[l].start).abs() < overlap_duration / 2.0
                {
                    current.push((k, l));
                    k += 1;
                    l += 1;
                }
                if current.len() > best_contiguous.len() {
                    best_contiguous = current;
                }
            }
        }
    }

    if best_contiguous.len() < enough_pairs {
        return Err(format!("no pairs exceeding {enough_pairs}"));
    }

    let a_start_idx = a.len() - overlap_a.len();
    let lcs_indices_a: Vec<usize> = best_contiguous
        .iter()
        .map(|(i, _)| a_start_idx + i)
        .collect();
    let lcs_indices_b: Vec<usize> = best_contiguous.iter().map(|(_, j)| *j).collect();

    let mut result = Vec::new();
    result.extend_from_slice(&a[..lcs_indices_a[0]]);

    for i in 0..best_contiguous.len() {
        let idx_a = lcs_indices_a[i];
        let idx_b = lcs_indices_b[i];

        result.push(a[idx_a].clone());

        if i < best_contiguous.len() - 1 {
            let next_idx_a = lcs_indices_a[i + 1];
            let next_idx_b = lcs_indices_b[i + 1];
            let gap_tokens_a = &a[idx_a + 1..next_idx_a];
            let gap_tokens_b = &b[idx_b + 1..next_idx_b];
            if gap_tokens_b.len() > gap_tokens_a.len() {
                result.extend_from_slice(gap_tokens_b);
            } else {
                result.extend_from_slice(gap_tokens_a);
            }
        }
    }

    result.extend_from_slice(&b[lcs_indices_b.last().unwrap() + 1..]);
    Ok(result)
}

#[must_use]
pub fn merge_longest_common_subsequence(
    a: &[AlignedToken],
    b: &[AlignedToken],
    overlap_duration: f64,
) -> Vec<AlignedToken> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let a_end_time = a.last().unwrap().end;
    let b_start_time = b.first().unwrap().start;

    if a_end_time <= b_start_time {
        let mut out = a.to_vec();
        out.extend_from_slice(b);
        return out;
    }

    let overlap_a: Vec<_> = a
        .iter()
        .filter(|t| t.end > b_start_time - overlap_duration)
        .cloned()
        .collect();
    let overlap_b: Vec<_> = b
        .iter()
        .filter(|t| t.start < a_end_time + overlap_duration)
        .cloned()
        .collect();

    if overlap_a.len() < 2 || overlap_b.len() < 2 {
        let cutoff_time = (a_end_time + b_start_time) / 2.0;
        let mut out: Vec<AlignedToken> =
            a.iter().filter(|t| t.end <= cutoff_time).cloned().collect();
        out.extend(b.iter().filter(|t| t.start >= cutoff_time).cloned());
        return out;
    }

    let mut dp = vec![vec![0usize; overlap_b.len() + 1]; overlap_a.len() + 1];
    for i in 1..=overlap_a.len() {
        for j in 1..=overlap_b.len() {
            if overlap_a[i - 1].id == overlap_b[j - 1].id
                && (overlap_a[i - 1].start - overlap_b[j - 1].start).abs() < overlap_duration / 2.0
            {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let mut lcs_pairs = Vec::new();
    let mut i = overlap_a.len();
    let mut j = overlap_b.len();
    while i > 0 && j > 0 {
        if overlap_a[i - 1].id == overlap_b[j - 1].id
            && (overlap_a[i - 1].start - overlap_b[j - 1].start).abs() < overlap_duration / 2.0
        {
            lcs_pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    lcs_pairs.reverse();

    if lcs_pairs.is_empty() {
        let cutoff_time = (a_end_time + b_start_time) / 2.0;
        let mut out: Vec<AlignedToken> =
            a.iter().filter(|t| t.end <= cutoff_time).cloned().collect();
        out.extend(b.iter().filter(|t| t.start >= cutoff_time).cloned());
        return out;
    }

    let a_start_idx = a.len() - overlap_a.len();
    let lcs_indices_a: Vec<usize> = lcs_pairs.iter().map(|(i, _)| a_start_idx + i).collect();
    let lcs_indices_b: Vec<usize> = lcs_pairs.iter().map(|(_, j)| *j).collect();

    let mut result = Vec::new();
    result.extend_from_slice(&a[..lcs_indices_a[0]]);
    for i in 0..lcs_pairs.len() {
        let idx_a = lcs_indices_a[i];
        let idx_b = lcs_indices_b[i];
        result.push(a[idx_a].clone());

        if i < lcs_pairs.len() - 1 {
            let next_idx_a = lcs_indices_a[i + 1];
            let next_idx_b = lcs_indices_b[i + 1];
            let gap_tokens_a = &a[idx_a + 1..next_idx_a];
            let gap_tokens_b = &b[idx_b + 1..next_idx_b];
            if gap_tokens_b.len() > gap_tokens_a.len() {
                result.extend_from_slice(gap_tokens_b);
            } else {
                result.extend_from_slice(gap_tokens_a);
            }
        }
    }

    result.extend_from_slice(&b[lcs_indices_b.last().unwrap() + 1..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(id: usize, text: &str, start: f64, dur: f64) -> AlignedToken {
        AlignedToken::new(id, text.to_string(), start, dur, 0.9)
    }

    #[test]
    fn aligned_token_end_computed() {
        let t = AlignedToken::new(0, "hi".to_string(), 1.0, 0.5, 0.9);
        assert!((t.end - 1.5).abs() < 1e-10);
    }

    #[test]
    fn aligned_sentence_computes_span() {
        let tokens = vec![tok(0, " hello", 0.0, 0.5), tok(1, " world", 0.6, 0.4)];
        let sentence = AlignedSentence::new("hello world".to_string(), tokens);
        assert!((sentence.start - 0.0).abs() < 1e-10);
        assert!((sentence.end - 1.0).abs() < 1e-10);
        assert!((sentence.duration - 1.0).abs() < 1e-10);
    }

    #[test]
    fn aligned_sentence_empty_tokens() {
        let sentence = AlignedSentence::new(String::new(), vec![]);
        assert!((sentence.confidence - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tokens_to_sentences_splits_on_period() {
        let tokens = vec![
            tok(0, " Hello", 0.0, 0.3),
            tok(1, ".", 0.3, 0.1),
            tok(2, " World", 0.5, 0.3),
        ];
        let config = SentenceConfig::default();
        let sentences = tokens_to_sentences(&tokens, &config);
        assert_eq!(sentences.len(), 2, "expected 2 sentences: {sentences:?}");
    }

    #[test]
    fn tokens_to_sentences_splits_on_question_mark() {
        let tokens = vec![
            tok(0, " Really", 0.0, 0.3),
            tok(1, "?", 0.3, 0.1),
            tok(2, " Yes", 0.5, 0.3),
        ];
        let sentences = tokens_to_sentences(&tokens, &SentenceConfig::default());
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn tokens_to_sentences_max_words() {
        let tokens = vec![
            tok(0, " a", 0.0, 0.1),
            tok(1, " b", 0.1, 0.1),
            tok(2, " c", 0.2, 0.1),
            tok(3, " d", 0.3, 0.1),
        ];
        let config = SentenceConfig {
            max_words: Some(2),
            ..Default::default()
        };
        let sentences = tokens_to_sentences(&tokens, &config);
        assert!(
            sentences.len() >= 2,
            "expected >=2 sentences for max_words=2"
        );
    }

    #[test]
    fn merge_contiguous_no_overlap() {
        let a = vec![tok(0, " a", 0.0, 0.5)];
        let b = vec![tok(1, " b", 1.0, 0.5)];
        let result = merge_longest_contiguous(&a, &b, 1.0).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, 0);
        assert_eq!(result[1].id, 1);
    }

    #[test]
    fn merge_contiguous_empty_inputs() {
        let empty: Vec<AlignedToken> = vec![];
        let a = vec![tok(0, " a", 0.0, 0.5)];
        assert_eq!(merge_longest_contiguous(&empty, &a, 1.0).unwrap().len(), 1);
        assert_eq!(merge_longest_contiguous(&a, &empty, 1.0).unwrap().len(), 1);
    }

    #[test]
    fn merge_lcs_no_overlap() {
        let a = vec![tok(0, " a", 0.0, 0.5)];
        let b = vec![tok(1, " b", 1.0, 0.5)];
        let result = merge_longest_common_subsequence(&a, &b, 1.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn merge_lcs_empty_inputs() {
        let empty: Vec<AlignedToken> = vec![];
        let a = vec![tok(0, " a", 0.0, 0.5)];
        assert_eq!(merge_longest_common_subsequence(&empty, &a, 1.0).len(), 1);
        assert_eq!(merge_longest_common_subsequence(&a, &empty, 1.0).len(), 1);
    }

    #[test]
    fn merge_contiguous_with_overlap() {
        // Tokens overlap in time, with matching IDs
        let a = vec![
            tok(0, " a", 0.0, 0.5),
            tok(1, " b", 0.5, 0.5),
            tok(2, " c", 1.0, 0.5),
            tok(3, " d", 1.5, 0.5),
        ];
        let b = vec![
            tok(2, " c", 1.0, 0.5),
            tok(3, " d", 1.5, 0.5),
            tok(4, " e", 2.0, 0.5),
        ];
        let result = merge_longest_contiguous(&a, &b, 2.0).unwrap();
        let ids: Vec<usize> = result.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn merge_lcs_with_overlap() {
        let a = vec![
            tok(0, " a", 0.0, 0.5),
            tok(1, " b", 0.5, 0.5),
            tok(2, " c", 1.0, 0.5),
            tok(3, " d", 1.5, 0.5),
        ];
        let b = vec![
            tok(2, " c", 1.0, 0.5),
            tok(3, " d", 1.5, 0.5),
            tok(4, " e", 2.0, 0.5),
        ];
        let result = merge_longest_common_subsequence(&a, &b, 2.0);
        let ids: Vec<usize> = result.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn merge_contiguous_fails_falls_back_to_lcs() {
        // Overlapping tokens where IDs don't form a contiguous match long
        // enough to pass the threshold, forcing LCS fallback.
        let a = vec![
            tok(0, " a", 0.0, 0.5),
            tok(1, " b", 0.5, 0.5),
            tok(2, " c", 1.0, 0.5),
            tok(99, " x", 1.5, 0.5), // divergent token
            tok(3, " d", 2.0, 0.5),
        ];
        let b = vec![
            tok(2, " c", 1.0, 0.5),
            tok(88, " y", 1.5, 0.5), // different divergent token
            tok(3, " d", 2.0, 0.5),
            tok(4, " e", 2.5, 0.5),
        ];
        // Contiguous should fail (max contiguous run is 1, threshold is 2)
        let contiguous_result = merge_longest_contiguous(&a, &b, 2.0);
        assert!(contiguous_result.is_err(), "contiguous should fail here");

        // LCS fallback should produce a valid merge
        let lcs_result = merge_longest_common_subsequence(&a, &b, 2.0);
        // Should contain tokens from both sides with IDs 2 and 3 as anchors
        let ids: Vec<usize> = lcs_result.iter().map(|t| t.id).collect();
        assert!(ids.contains(&0), "should have token 0 from a");
        assert!(ids.contains(&4), "should have token 4 from b");
        assert!(ids.contains(&2), "should have shared token 2");
        assert!(ids.contains(&3), "should have shared token 3");
    }

    #[test]
    fn merge_contiguous_few_overlap_uses_cutoff() {
        // When overlap has < 2 tokens, the cutoff-time heuristic is used
        let a = vec![tok(0, " a", 0.0, 0.5), tok(1, " b", 0.5, 0.5)];
        let b = vec![tok(2, " c", 0.8, 0.5), tok(3, " d", 1.5, 0.5)];
        let result = merge_longest_contiguous(&a, &b, 0.1).unwrap();
        // Should have tokens from both sides separated at cutoff
        assert!(!result.is_empty());
    }

    #[test]
    fn iter_tokens_matches_tokens() {
        let sentences = vec![
            AlignedSentence::new("Hello.".to_string(), vec![tok(0, "Hello.", 0.0, 0.5)]),
            AlignedSentence::new(" World.".to_string(), vec![tok(1, " World.", 1.0, 0.5)]),
        ];
        let result = sentences_to_result(&sentences);
        let owned: Vec<usize> = result.tokens().iter().map(|t| t.id).collect();
        let refs: Vec<usize> = result.iter_tokens().map(|t| t.id).collect();
        assert_eq!(owned, refs);
    }

    #[test]
    fn sentences_to_result_concatenates_text() {
        let sentences = vec![
            AlignedSentence::new("Hello.".to_string(), vec![tok(0, "Hello.", 0.0, 0.5)]),
            AlignedSentence::new(" World.".to_string(), vec![tok(1, " World.", 1.0, 0.5)]),
        ];
        let result = sentences_to_result(&sentences);
        assert_eq!(result.text, "Hello. World.");
    }
}
