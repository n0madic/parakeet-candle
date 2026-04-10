pub fn decode(tokens: &[usize], vocabulary: &[String]) -> String {
    tokens
        .iter()
        .filter_map(|&id| vocabulary.get(id))
        .map(|s| s.replace('▁', " "))
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vec<String> {
        vec![
            "▁hello".to_string(),
            "▁world".to_string(),
            "▁foo".to_string(),
        ]
    }

    #[test]
    fn decode_basic() {
        assert_eq!(decode(&[0, 1], &vocab()), " hello world");
    }

    #[test]
    fn decode_replaces_sentencepiece_space() {
        assert_eq!(decode(&[0], &vocab()), " hello");
    }

    #[test]
    fn decode_out_of_bounds_skipped() {
        assert_eq!(decode(&[0, 99, 1], &vocab()), " hello world");
    }

    #[test]
    fn decode_empty() {
        assert_eq!(decode(&[], &vocab()), "");
    }
}
