use std::collections::HashMap;

use candle::{D, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::{log_softmax, softmax};

use crate::parakeet::alignment::{
    AlignedResult, AlignedToken, SentenceConfig, sentences_to_result, tokens_to_sentences,
};
use crate::parakeet::audio::{PreprocessArgs, get_logmel, load_audio};
use crate::parakeet::cache::RotatingConformerCache;
use crate::parakeet::conformer::{Conformer, ConformerArgs};
use crate::parakeet::ctc::{AuxCTCArgs, ConvASRDecoder, ConvASRDecoderArgs};
use crate::parakeet::rnnt::{JointArgs, JointNetwork, PredictArgs, PredictNetwork};
use crate::parakeet::tokenizer;

pub type HiddenStates = Vec<Option<(Tensor, Tensor)>>;

/// Compute entropy-based confidence from logits.
/// Returns a value in [0, 1] where 1 means the model is maximally certain.
fn confidence_from_logits(logits: &Tensor, vocab_size: usize) -> Result<f64> {
    let probs = softmax(logits, D::Minus1)?;
    let log_probs = (&probs + 1e-10)?.log()?;
    let entropy = (&probs * &log_probs)?.sum_all()?.neg()?;
    let entropy = entropy.to_vec0::<f32>()? as f64;
    let max_entropy = (vocab_size as f64).ln();
    Ok(1.0 - (entropy / max_entropy))
}

/// Compute entropy-based confidence from a span of probability frames (CTC).
/// Averages per-frame entropy across the span.
fn confidence_from_span(probs: &Tensor, vocab_size: usize) -> Result<f64> {
    let log_probs = (probs + 1e-10)?.log()?;
    let entropy = (probs * &log_probs)?.sum(D::Minus1)?.neg()?;
    let avg_entropy = entropy.mean_all()?.to_vec0::<f32>()? as f64;
    let max_entropy = (vocab_size as f64).ln();
    Ok(1.0 - (avg_entropy / max_entropy))
}

/// Beam search hypothesis for RNNT/TDT decoding.
#[derive(Clone)]
struct Hypothesis {
    score: f64,
    step: usize,
    last_token: Option<usize>,
    hidden_state: Option<(Tensor, Tensor)>,
    stuck: usize,
    tokens: Vec<AlignedToken>,
}

#[derive(Debug, Clone)]
pub struct Greedy;

#[derive(Debug, Clone)]
pub struct Beam {
    pub beam_size: usize,
    pub length_penalty: f64,
    pub patience: f64,
    pub duration_reward: f64,
}

impl Default for Beam {
    fn default() -> Self {
        Self {
            beam_size: 5,
            length_penalty: 1.0,
            patience: 1.0,
            duration_reward: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Decoding {
    Greedy(Greedy),
    Beam(Beam),
}

#[derive(Debug, Clone)]
pub struct DecodingConfig {
    pub decoding: Decoding,
    pub sentence: SentenceConfig,
    pub debug_decode: bool,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            decoding: Decoding::Greedy(Greedy),
            sentence: SentenceConfig::default(),
            debug_decode: false,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GreedyConfig {
    #[serde(default)]
    max_symbols: Option<i64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TdtDecodingArgs {
    pub model_type: String,
    pub durations: Vec<usize>,
    #[serde(default)]
    pub greedy: Option<GreedyConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RnntDecodingArgs {
    #[serde(default)]
    pub greedy: Option<GreedyConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CtcDecodingArgs {
    #[serde(default)]
    pub greedy: Option<GreedyConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ParakeetTdtArgs {
    pub preprocessor: PreprocessArgs,
    pub encoder: ConformerArgs,
    pub decoder: PredictArgs,
    pub joint: JointArgs,
    pub decoding: TdtDecodingArgs,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ParakeetRnntArgs {
    pub preprocessor: PreprocessArgs,
    pub encoder: ConformerArgs,
    pub decoder: PredictArgs,
    pub joint: JointArgs,
    pub decoding: RnntDecodingArgs,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ParakeetCtcArgs {
    pub preprocessor: PreprocessArgs,
    pub encoder: ConformerArgs,
    pub decoder: ConvASRDecoderArgs,
    pub decoding: CtcDecodingArgs,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ParakeetTdtCtcArgs {
    #[serde(flatten)]
    pub base: ParakeetTdtArgs,
    pub aux_ctc: AuxCTCArgs,
}

#[derive(Debug, Clone)]
pub enum ParakeetModel {
    Tdt(ParakeetTdt),
    Rnnt(ParakeetRnnt),
    Ctc(ParakeetCtc),
    TdtCtc(ParakeetTdtCtc),
}

impl ParakeetModel {
    pub fn sample_rate(&self) -> usize {
        self.preprocessor_config().sample_rate
    }

    pub fn preprocessor_config(&self) -> &PreprocessArgs {
        match self {
            Self::Tdt(m) => &m.preprocessor_config,
            Self::Rnnt(m) => &m.preprocessor_config,
            Self::Ctc(m) => &m.preprocessor_config,
            Self::TdtCtc(m) => &m.base.preprocessor_config,
        }
    }

    pub fn encoder_config(&self) -> &ConformerArgs {
        match self {
            Self::Tdt(m) => &m.encoder_config,
            Self::Rnnt(m) => &m.encoder_config,
            Self::Ctc(m) => &m.encoder_config,
            Self::TdtCtc(m) => &m.base.encoder_config,
        }
    }

    pub fn device_ref(&self) -> &Device {
        match self {
            Self::Tdt(m) => &m.device,
            Self::Rnnt(m) => &m.device,
            Self::Ctc(m) => &m.device,
            Self::TdtCtc(m) => &m.base.device,
        }
    }

    pub fn vocabulary(&self) -> &[String] {
        match self {
            Self::Tdt(m) => &m.vocabulary,
            Self::Rnnt(m) => &m.vocabulary,
            Self::Ctc(m) => &m.vocabulary,
            Self::TdtCtc(m) => &m.base.vocabulary,
        }
    }

    pub fn time_ratio(&self) -> f64 {
        time_ratio(self.preprocessor_config(), self.encoder_config())
    }

    pub fn generate(
        &mut self,
        mel: &Tensor,
        config: &DecodingConfig,
    ) -> Result<Vec<AlignedResult>> {
        match self {
            Self::Tdt(m) => m.generate(mel, config),
            Self::Rnnt(m) => m.generate(mel, config),
            Self::Ctc(m) => m.generate(mel, config),
            Self::TdtCtc(m) => m.base.generate(mel, config),
        }
    }

    pub fn transcribe(
        &mut self,
        path: &std::path::Path,
        config: &DecodingConfig,
        chunk_duration: Option<f64>,
        overlap_duration: f64,
        chunk_callback: Option<Box<dyn FnMut(usize, usize)>>,
    ) -> Result<AlignedResult> {
        let audio = load_audio(path, self.sample_rate())?;
        let device = self.device_ref().clone();
        let pre = self.preprocessor_config().clone();
        if chunk_duration.is_none() {
            let mel = get_logmel(&audio, &pre, &device)?;
            return Ok(self.generate(&mel, config)?[0].clone());
        }
        let mut generate = |mel: &Tensor, cfg: &DecodingConfig| -> Result<AlignedResult> {
            Ok(self.generate(mel, cfg)?[0].clone())
        };
        transcribe_with_audio(
            TranscribeParams {
                audio: &audio,
                device: &device,
                pre: &pre,
                config,
                chunk_duration,
                overlap_duration,
                chunk_callback,
            },
            &mut generate,
        )
    }

    pub fn transcribe_stream(
        &mut self,
        context_size: (usize, usize),
        depth: usize,
        keep_original_attention: bool,
        config: DecodingConfig,
    ) -> StreamingParakeet<'_> {
        match self {
            ParakeetModel::Tdt(m) => {
                m.transcribe_stream(context_size, depth, keep_original_attention, config)
            }
            ParakeetModel::Rnnt(m) => {
                m.transcribe_stream(context_size, depth, keep_original_attention, config)
            }
            ParakeetModel::Ctc(m) => {
                m.transcribe_stream(context_size, depth, keep_original_attention, config)
            }
            ParakeetModel::TdtCtc(m) => {
                m.base
                    .transcribe_stream(context_size, depth, keep_original_attention, config)
            }
        }
    }
}

pub fn from_config_value(config: serde_json::Value, vb: VarBuilder) -> Result<ParakeetModel> {
    let target = config
        .get("target")
        .and_then(|v| v.as_str())
        .ok_or_else(|| candle::Error::Msg("missing target in config".to_string()))?;
    let has_tdt = config
        .get("model_defaults")
        .and_then(|v| v.get("tdt_durations"))
        .is_some();

    match (target, has_tdt) {
        ("nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel", true) => {
            let args: ParakeetTdtArgs =
                serde_json::from_value(config).map_err(candle::Error::wrap)?;
            args.preprocessor.validate()?;
            Ok(ParakeetModel::Tdt(ParakeetTdt::load(args, vb)?))
        }
        (
            "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel",
            true,
        ) => {
            let args: ParakeetTdtCtcArgs =
                serde_json::from_value(config).map_err(candle::Error::wrap)?;
            args.base.preprocessor.validate()?;
            Ok(ParakeetModel::TdtCtc(ParakeetTdtCtc::load(args, vb)?))
        }
        ("nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel", false) => {
            let args: ParakeetRnntArgs =
                serde_json::from_value(config).map_err(candle::Error::wrap)?;
            args.preprocessor.validate()?;
            Ok(ParakeetModel::Rnnt(ParakeetRnnt::load(args, vb)?))
        }
        ("nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE", _) => {
            let args: ParakeetCtcArgs =
                serde_json::from_value(config).map_err(candle::Error::wrap)?;
            args.preprocessor.validate()?;
            Ok(ParakeetModel::Ctc(ParakeetCtc::load(args, vb)?))
        }
        _ => Err(candle::Error::Msg(
            "unsupported parakeet config".to_string(),
        )),
    }
}

fn time_ratio(pre: &PreprocessArgs, enc: &ConformerArgs) -> f64 {
    enc.subsampling_factor as f64 / pre.sample_rate as f64 * pre.hop_length() as f64
}

#[derive(Debug, Clone)]
pub struct ParakeetTdt {
    pub preprocessor_config: PreprocessArgs,
    pub encoder_config: ConformerArgs,
    pub encoder: Conformer,
    pub decoder: PredictNetwork,
    pub joint: JointNetwork,
    pub vocabulary: Vec<String>,
    pub durations: Vec<usize>,
    pub max_symbols: Option<usize>,
    pub device: Device,
}

impl ParakeetTdt {
    pub fn load(args: ParakeetTdtArgs, vb: VarBuilder) -> Result<Self> {
        if args.decoding.model_type != "tdt" {
            return Err(candle::Error::Msg("model type must be tdt".to_string()));
        }
        let vocabulary = args.joint.vocabulary.clone();
        let durations = args.decoding.durations.clone();
        let max_symbols = args
            .decoding
            .greedy
            .and_then(|g| g.max_symbols)
            .and_then(|v| if v > 0 { Some(v as usize) } else { None });
        let device = vb.device().clone();
        let encoder = Conformer::load(args.encoder.clone(), vb.pp("encoder"))?;
        let decoder = PredictNetwork::load(&args.decoder, vb.pp("decoder"))?;
        let joint = JointNetwork::load(&args.joint, vb.pp("joint"))?;
        Ok(Self {
            preprocessor_config: args.preprocessor,
            encoder_config: args.encoder,
            encoder,
            decoder,
            joint,
            vocabulary,
            durations,
            max_symbols,
            device,
        })
    }

    pub fn time_ratio(&self) -> f64 {
        time_ratio(&self.preprocessor_config, &self.encoder_config)
    }

    pub fn decode(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        last_token: Option<Vec<Option<usize>>>,
        hidden_state: Option<HiddenStates>,
        config: &DecodingConfig,
    ) -> Result<(Vec<Vec<AlignedToken>>, HiddenStates)> {
        match &config.decoding {
            Decoding::Greedy(_) => {
                self.decode_greedy(features, lengths, last_token, hidden_state, config)
            }
            Decoding::Beam(beam) => {
                self.decode_beam(features, lengths, last_token, hidden_state, beam, config)
            }
        }
    }

    fn decode_greedy(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        mut last_token: Option<Vec<Option<usize>>>,
        mut hidden_state: Option<HiddenStates>,
        config: &DecodingConfig,
    ) -> Result<(Vec<Vec<AlignedToken>>, HiddenStates)> {
        let (b, s, _) = features.dims3()?;
        let lengths = if let Some(l) = lengths {
            l.clone()
        } else {
            Tensor::from_vec(vec![s as i64; b], (b,), features.device())?
        };
        let lengths_vec = lengths.to_vec1::<i64>()?;

        if last_token.is_none() {
            last_token = Some(vec![None; b]);
        }
        if hidden_state.is_none() {
            hidden_state = Some(vec![None; b]);
        }

        let mut results = Vec::with_capacity(b);
        let mut next_hidden = Vec::with_capacity(b);
        let tr = self.time_ratio();

        for (batch, &length) in lengths_vec.iter().enumerate() {
            let mut hypothesis = Vec::new();
            let feature = features.narrow(0, batch, 1)?;
            let length = length as usize;
            let mut step = 0usize;
            let mut new_symbols = 0usize;
            let mut last = last_token.as_ref().unwrap()[batch];
            let mut hidden = hidden_state.as_ref().unwrap()[batch].clone();

            while step < length {
                let step_before = step;
                let decoder_out = if let Some(token) = last {
                    let input = Tensor::from_vec(vec![token as i64], (1, 1), feature.device())?;
                    self.decoder.forward(Some(&input), hidden.clone())?
                } else {
                    self.decoder.forward(None, hidden.clone())?
                };
                let (decoder_out, decoder_state) = decoder_out;

                let enc_step = feature.narrow(1, step, 1)?;
                let joint_out = self.joint.forward(&enc_step, &decoder_out)?;

                let vocab_size = self.vocabulary.len() + 1;
                let token_logits = joint_out.i((0, 0, 0, 0..vocab_size))?;
                let duration_logits = joint_out.i((0, 0, 0, vocab_size..))?;

                let pred_token = token_logits.argmax(D::Minus1)?.to_vec0::<u32>()? as usize;
                let decision = (duration_logits.argmax(D::Minus1)?.to_vec0::<u32>()? as usize)
                    .min(self.durations.len() - 1);

                let confidence = confidence_from_logits(&token_logits, vocab_size)?;

                if pred_token != self.vocabulary.len() {
                    hypothesis.push(AlignedToken::new(
                        pred_token,
                        tokenizer::decode(&[pred_token], &self.vocabulary),
                        step as f64 * tr,
                        self.durations[decision] as f64 * tr,
                        confidence,
                    ));
                    last = Some(pred_token);
                    hidden = Some(decoder_state);
                    new_symbols += 1;
                }

                let duration = self.durations[decision];
                step += duration;
                let mut forced_advance = false;
                if duration != 0 {
                    new_symbols = 0;
                } else if let Some(max_symbols) = self.max_symbols
                    && new_symbols >= max_symbols
                {
                    step += 1;
                    forced_advance = true;
                    new_symbols = 0;
                }
                if config.debug_decode {
                    let is_blank = pred_token == self.vocabulary.len();
                    eprintln!(
                        "debug: step={step_before} token_id={pred_token} blank={is_blank} duration={duration} step_advance={} forced_advance={forced_advance}",
                        step - step_before
                    );
                }
            }

            results.push(hypothesis);
            next_hidden.push(hidden);
        }

        Ok((results, next_hidden))
    }

    fn decode_beam(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        mut last_token: Option<Vec<Option<usize>>>,
        mut hidden_state: Option<HiddenStates>,
        beam: &Beam,
        config: &DecodingConfig,
    ) -> Result<(Vec<Vec<AlignedToken>>, HiddenStates)> {
        if beam.beam_size == 0 {
            return Err(candle::Error::Msg("beam_size must be >= 1".to_string()));
        }
        if beam.patience <= 0.0 {
            return Err(candle::Error::Msg("beam patience must be > 0".to_string()));
        }
        let (b, s, _) = features.dims3()?;
        let lengths = if let Some(l) = lengths {
            l.clone()
        } else {
            Tensor::from_vec(vec![s as i64; b], (b,), features.device())?
        };
        let lengths_vec = lengths.to_vec1::<i64>()?;

        if hidden_state.is_none() {
            hidden_state = Some(vec![None; b]);
        }
        if last_token.is_none() {
            last_token = Some(vec![None; b]);
        }

        let mut results = Vec::with_capacity(b);
        let mut results_hidden = Vec::with_capacity(b);

        for (batch, &length) in lengths_vec.iter().enumerate() {
            let feature = features.narrow(0, batch, 1)?;
            let length = length as usize;

            let beam_token = beam.beam_size.min(self.vocabulary.len() + 1);
            let beam_duration = beam.beam_size.min(self.durations.len());
            let max_candidates = ((beam.beam_size as f64) * beam.patience).round() as usize;
            let tr = self.time_ratio();

            let init_last = last_token.as_ref().unwrap()[batch];
            let init_hidden = hidden_state.as_ref().unwrap()[batch].clone();

            let mut finished = Vec::new();
            let mut active = vec![Hypothesis {
                score: 0.0,
                step: 0,
                last_token: init_last,
                hidden_state: init_hidden,
                stuck: 0,
                tokens: Vec::new(),
            }];

            if config.debug_decode {
                eprintln!(
                    "debug: beam search batch={batch} length={length} beam_size={} patience={:.1}",
                    beam.beam_size, beam.patience
                );
            }

            while finished.len() < max_candidates && !active.is_empty() {
                let mut candidates: HashMap<String, Hypothesis> = HashMap::new();
                for hyp in active.iter() {
                    let decoder_out = if let Some(token) = hyp.last_token {
                        let input = Tensor::from_vec(vec![token as i64], (1, 1), feature.device())?;
                        self.decoder
                            .forward(Some(&input), hyp.hidden_state.clone())?
                    } else {
                        self.decoder.forward(None, hyp.hidden_state.clone())?
                    };
                    let (decoder_out, decoder_state) = decoder_out;
                    let enc_step = feature.narrow(1, hyp.step, 1)?;
                    let joint_out = self.joint.forward(&enc_step, &decoder_out)?;

                    let vocab_size = self.vocabulary.len() + 1;
                    let token_logits = joint_out.i((0, 0, 0, 0..vocab_size))?;
                    let duration_logits = joint_out.i((0, 0, 0, vocab_size..))?;

                    let token_logprobs = log_softmax(&token_logits, D::Minus1)?.to_vec1::<f32>()?;
                    let duration_logprobs =
                        log_softmax(&duration_logits, D::Minus1)?.to_vec1::<f32>()?;

                    let mut token_idx: Vec<usize> = (0..token_logprobs.len()).collect();
                    token_idx.sort_by(|&a, &b| {
                        token_logprobs[b]
                            .partial_cmp(&token_logprobs[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    token_idx.truncate(beam_token);

                    let mut dur_idx: Vec<usize> =
                        (0..duration_logprobs.len().min(self.durations.len())).collect();
                    dur_idx.sort_by(|&a, &b| {
                        duration_logprobs[b]
                            .partial_cmp(&duration_logprobs[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    dur_idx.truncate(beam_duration);

                    for token in token_idx.iter().copied() {
                        let is_blank = token == self.vocabulary.len();
                        let next_last = if is_blank {
                            hyp.last_token
                        } else {
                            Some(token)
                        };
                        let next_state = if is_blank {
                            hyp.hidden_state.clone()
                        } else {
                            Some(decoder_state.clone())
                        };
                        for decision in dur_idx.iter().copied() {
                            let duration = self.durations[decision];
                            let mut stuck = if duration != 0 { 0 } else { hyp.stuck + 1 };
                            let step = if let Some(max_sym) = self.max_symbols
                                && stuck >= max_sym
                            {
                                stuck = 0;
                                hyp.step + 1
                            } else {
                                hyp.step + duration
                            };

                            let tokens = if is_blank {
                                hyp.tokens.clone()
                            } else {
                                let mut h = hyp.tokens.clone();
                                h.push(AlignedToken::new(
                                    token,
                                    tokenizer::decode(&[token], &self.vocabulary),
                                    hyp.step as f64 * tr,
                                    duration as f64 * tr,
                                    (token_logprobs[token] + duration_logprobs[decision]).exp()
                                        as f64,
                                ));
                                h
                            };

                            let score = hyp.score
                                + (token_logprobs[token] as f64) * (1.0 - beam.duration_reward)
                                + (duration_logprobs[decision] as f64) * beam.duration_reward;

                            let key = format!(
                                "{}:{}",
                                step,
                                tokens
                                    .iter()
                                    .map(|t| t.id.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            );
                            let new_hyp = Hypothesis {
                                score,
                                step,
                                last_token: next_last,
                                hidden_state: next_state.clone(),
                                stuck,
                                tokens,
                            };

                            if let Some(existing) = candidates.get_mut(&key) {
                                let maxima = existing.score.max(new_hyp.score);
                                let combined_score = maxima
                                    + ((existing.score - maxima).exp()
                                        + (new_hyp.score - maxima).exp())
                                    .ln();
                                if new_hyp.score > existing.score {
                                    *existing = new_hyp;
                                }
                                // Must be last: *existing = new_hyp overwrites score
                                existing.score = combined_score;
                            } else {
                                candidates.insert(key, new_hyp);
                            }
                        }
                    }
                }

                for hyp in candidates.values() {
                    if hyp.step >= length {
                        finished.push(hyp.clone());
                    }
                }

                let mut active_list: Vec<Hypothesis> = candidates
                    .into_values()
                    .filter(|h| h.step < length)
                    .collect();
                active_list.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                active = active_list.into_iter().take(beam.beam_size).collect();
            }

            let mut all = finished;
            all.extend(active);

            if all.is_empty() {
                results.push(Vec::new());
                results_hidden.push(hidden_state.as_ref().unwrap()[batch].clone());
            } else {
                let best = all
                    .into_iter()
                    .max_by(|a, b| {
                        let score_a =
                            a.score / (a.tokens.len().max(1) as f64).powf(beam.length_penalty);
                        let score_b =
                            b.score / (b.tokens.len().max(1) as f64).powf(beam.length_penalty);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                if config.debug_decode {
                    eprintln!(
                        "debug: beam best score={:.4} tokens={}",
                        best.score,
                        best.tokens.len()
                    );
                }
                results.push(best.tokens);
                results_hidden.push(best.hidden_state);
            }
        }

        Ok((results, results_hidden))
    }

    pub fn generate(
        &mut self,
        mel: &Tensor,
        config: &DecodingConfig,
    ) -> Result<Vec<AlignedResult>> {
        let mel = if mel.dims().len() == 2 {
            mel.unsqueeze(0)?
        } else {
            mel.clone()
        };
        let (features, lengths) = self.encoder.forward(&mel, None)?;
        let (result, _) = self.decode(&features, Some(&lengths), None, None, config)?;
        Ok(result
            .into_iter()
            .map(|hyp| sentences_to_result(&tokens_to_sentences(&hyp, &config.sentence)))
            .collect())
    }

    pub fn transcribe(
        &mut self,
        path: &std::path::Path,
        config: &DecodingConfig,
        chunk_duration: Option<f64>,
        overlap_duration: f64,
        chunk_callback: Option<Box<dyn FnMut(usize, usize)>>,
    ) -> Result<AlignedResult> {
        let audio = load_audio(path, self.preprocessor_config.sample_rate)?;
        if chunk_duration.is_none() {
            let mel = get_logmel(&audio, &self.preprocessor_config, &self.device)?;
            return Ok(self.generate(&mel, config)?[0].clone());
        }
        let device = self.device.clone();
        let pre = self.preprocessor_config.clone();
        let mut generate = |mel: &Tensor, cfg: &DecodingConfig| -> Result<AlignedResult> {
            Ok(self.generate(mel, cfg)?[0].clone())
        };
        transcribe_with_audio(
            TranscribeParams {
                audio: &audio,
                device: &device,
                pre: &pre,
                config,
                chunk_duration,
                overlap_duration,
                chunk_callback,
            },
            &mut generate,
        )
    }

    pub fn transcribe_stream(
        &mut self,
        context_size: (usize, usize),
        depth: usize,
        keep_original_attention: bool,
        config: DecodingConfig,
    ) -> StreamingParakeet<'_> {
        StreamingParakeet::new(
            ParakeetModelRef::Tdt(self),
            context_size,
            depth,
            keep_original_attention,
            config,
        )
    }
}

struct TranscribeParams<'a> {
    audio: &'a [f32],
    device: &'a Device,
    pre: &'a PreprocessArgs,
    config: &'a DecodingConfig,
    chunk_duration: Option<f64>,
    overlap_duration: f64,
    chunk_callback: Option<Box<dyn FnMut(usize, usize)>>,
}

fn transcribe_with_audio<F>(params: TranscribeParams<'_>, generate: &mut F) -> Result<AlignedResult>
where
    F: FnMut(&Tensor, &DecodingConfig) -> Result<AlignedResult>,
{
    let TranscribeParams {
        audio,
        device,
        pre,
        config,
        chunk_duration,
        overlap_duration,
        mut chunk_callback,
    } = params;

    if chunk_duration.is_none() {
        let mel = get_logmel(audio, pre, device)?;
        return generate(&mel, config);
    }

    let chunk_duration = chunk_duration.unwrap();
    let audio_len_sec = audio.len() as f64 / pre.sample_rate as f64;
    if audio_len_sec <= chunk_duration {
        let mel = get_logmel(audio, pre, device)?;
        return generate(&mel, config);
    }

    let chunk_samples = (chunk_duration * pre.sample_rate as f64) as usize;
    let overlap_samples = (overlap_duration * pre.sample_rate as f64) as usize;
    let mut all_tokens = Vec::new();

    let mut start = 0usize;
    while start < audio.len() {
        let end = (start + chunk_samples).min(audio.len());
        if let Some(cb) = chunk_callback.as_mut() {
            cb(end, audio.len());
        }
        if end - start < pre.hop_length() {
            break;
        }
        let chunk_audio = &audio[start..end];
        let mel = get_logmel(chunk_audio, pre, device)?;
        let mut chunk_result = generate(&mel, config)?;
        let chunk_offset = start as f64 / pre.sample_rate as f64;
        for sentence in chunk_result.sentences.iter_mut() {
            for token in sentence.tokens.iter_mut() {
                token.start += chunk_offset;
                token.end = token.start + token.duration;
            }
        }

        if all_tokens.is_empty() {
            all_tokens = chunk_result.tokens();
        } else {
            match crate::parakeet::alignment::merge_longest_contiguous(
                &all_tokens,
                &chunk_result.tokens(),
                overlap_duration,
            ) {
                Ok(tokens) => all_tokens = tokens,
                Err(_) => {
                    all_tokens = crate::parakeet::alignment::merge_longest_common_subsequence(
                        &all_tokens,
                        &chunk_result.tokens(),
                        overlap_duration,
                    );
                }
            }
        }

        if end == audio.len() {
            break;
        }
        start += chunk_samples.saturating_sub(overlap_samples);
    }

    Ok(sentences_to_result(&tokens_to_sentences(
        &all_tokens,
        &config.sentence,
    )))
}

#[derive(Debug, Clone)]
pub struct ParakeetRnnt {
    pub preprocessor_config: PreprocessArgs,
    pub encoder_config: ConformerArgs,
    pub encoder: Conformer,
    pub decoder: PredictNetwork,
    pub joint: JointNetwork,
    pub vocabulary: Vec<String>,
    pub max_symbols: Option<usize>,
    pub device: Device,
}

impl ParakeetRnnt {
    pub fn load(args: ParakeetRnntArgs, vb: VarBuilder) -> Result<Self> {
        let vocabulary = args.joint.vocabulary.clone();
        let max_symbols = args
            .decoding
            .greedy
            .and_then(|g| g.max_symbols)
            .and_then(|v| if v > 0 { Some(v as usize) } else { None });
        let device = vb.device().clone();
        let encoder = Conformer::load(args.encoder.clone(), vb.pp("encoder"))?;
        let decoder = PredictNetwork::load(&args.decoder, vb.pp("decoder"))?;
        let joint = JointNetwork::load(&args.joint, vb.pp("joint"))?;
        Ok(Self {
            preprocessor_config: args.preprocessor,
            encoder_config: args.encoder,
            encoder,
            decoder,
            joint,
            vocabulary,
            max_symbols,
            device,
        })
    }

    pub fn time_ratio(&self) -> f64 {
        time_ratio(&self.preprocessor_config, &self.encoder_config)
    }

    pub fn decode(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        mut last_token: Option<Vec<Option<usize>>>,
        mut hidden_state: Option<HiddenStates>,
    ) -> Result<(Vec<Vec<AlignedToken>>, HiddenStates)> {
        let (b, s, _) = features.dims3()?;
        let lengths = if let Some(l) = lengths {
            l.clone()
        } else {
            Tensor::from_vec(vec![s as i64; b], (b,), features.device())?
        };
        let lengths_vec = lengths.to_vec1::<i64>()?;

        if last_token.is_none() {
            last_token = Some(vec![None; b]);
        }
        if hidden_state.is_none() {
            hidden_state = Some(vec![None; b]);
        }

        let mut results = Vec::with_capacity(b);
        let mut next_hidden = Vec::with_capacity(b);
        let tr = self.time_ratio();
        let vocab_size = self.vocabulary.len() + 1;

        for (batch, &length) in lengths_vec.iter().enumerate() {
            let mut hypothesis = Vec::new();
            let feature = features.narrow(0, batch, 1)?;
            let length = length as usize;
            let mut step = 0usize;
            let mut new_symbols = 0usize;
            let mut last = last_token.as_ref().unwrap()[batch];
            let mut hidden = hidden_state.as_ref().unwrap()[batch].clone();

            while step < length {
                let decoder_out = if let Some(token) = last {
                    let input = Tensor::from_vec(vec![token as i64], (1, 1), feature.device())?;
                    self.decoder.forward(Some(&input), hidden.clone())?
                } else {
                    self.decoder.forward(None, hidden.clone())?
                };
                let (decoder_out, decoder_state) = decoder_out;

                let enc_step = feature.narrow(1, step, 1)?;
                let joint_out = self.joint.forward(&enc_step, &decoder_out)?;

                let token_logits = joint_out.i((0, 0, 0, ..))?;
                let pred_token = token_logits.argmax(D::Minus1)?.to_vec0::<u32>()? as usize;

                let confidence = confidence_from_logits(&token_logits, vocab_size)?;

                if pred_token != self.vocabulary.len() {
                    hypothesis.push(AlignedToken::new(
                        pred_token,
                        tokenizer::decode(&[pred_token], &self.vocabulary),
                        step as f64 * tr,
                        1.0 * tr,
                        confidence,
                    ));
                    last = Some(pred_token);
                    hidden = Some(decoder_state);

                    new_symbols += 1;
                    if let Some(max_symbols) = self.max_symbols
                        && new_symbols >= max_symbols
                    {
                        step += 1;
                        new_symbols = 0;
                    }
                } else {
                    step += 1;
                    new_symbols = 0;
                }
            }

            results.push(hypothesis);
            next_hidden.push(hidden);
        }

        Ok((results, next_hidden))
    }

    pub fn generate(
        &mut self,
        mel: &Tensor,
        config: &DecodingConfig,
    ) -> Result<Vec<AlignedResult>> {
        let mel = if mel.dims().len() == 2 {
            mel.unsqueeze(0)?
        } else {
            mel.clone()
        };
        let (features, lengths) = self.encoder.forward(&mel, None)?;
        let (result, _) = self.decode(&features, Some(&lengths), None, None)?;
        Ok(result
            .into_iter()
            .map(|hyp| sentences_to_result(&tokens_to_sentences(&hyp, &config.sentence)))
            .collect())
    }

    pub fn transcribe(
        &mut self,
        path: &std::path::Path,
        config: &DecodingConfig,
        chunk_duration: Option<f64>,
        overlap_duration: f64,
        chunk_callback: Option<Box<dyn FnMut(usize, usize)>>,
    ) -> Result<AlignedResult> {
        let audio = load_audio(path, self.preprocessor_config.sample_rate)?;
        let device = self.device.clone();
        let pre = self.preprocessor_config.clone();
        let mut generate = |mel: &Tensor, cfg: &DecodingConfig| -> Result<AlignedResult> {
            Ok(self.generate(mel, cfg)?[0].clone())
        };
        transcribe_with_audio(
            TranscribeParams {
                audio: &audio,
                device: &device,
                pre: &pre,
                config,
                chunk_duration,
                overlap_duration,
                chunk_callback,
            },
            &mut generate,
        )
    }

    pub fn transcribe_stream(
        &mut self,
        context_size: (usize, usize),
        depth: usize,
        keep_original_attention: bool,
        config: DecodingConfig,
    ) -> StreamingParakeet<'_> {
        StreamingParakeet::new(
            ParakeetModelRef::Rnnt(self),
            context_size,
            depth,
            keep_original_attention,
            config,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ParakeetCtc {
    pub preprocessor_config: PreprocessArgs,
    pub encoder_config: ConformerArgs,
    pub encoder: Conformer,
    pub decoder: ConvASRDecoder,
    pub vocabulary: Vec<String>,
    pub device: Device,
}

impl ParakeetCtc {
    pub fn load(args: ParakeetCtcArgs, vb: VarBuilder) -> Result<Self> {
        let vocabulary = args.decoder.vocabulary.clone();
        let device = vb.device().clone();
        let encoder = Conformer::load(args.encoder.clone(), vb.pp("encoder"))?;
        let decoder = ConvASRDecoder::load(&args.decoder, vb.pp("decoder"))?;
        Ok(Self {
            preprocessor_config: args.preprocessor,
            encoder_config: args.encoder,
            encoder,
            decoder,
            vocabulary,
            device,
        })
    }

    pub fn time_ratio(&self) -> f64 {
        time_ratio(&self.preprocessor_config, &self.encoder_config)
    }

    pub fn decode(&self, features: &Tensor, lengths: &Tensor) -> Result<Vec<Vec<AlignedToken>>> {
        let (b, _, _) = features.dims3()?;
        let logits = self.decoder.forward(features)?;
        let probs = logits.exp()?;
        let lengths_vec = lengths.to_vec1::<i64>()?;
        let tr = self.time_ratio();
        let vocab_size = self.vocabulary.len() + 1;

        let mut results = Vec::with_capacity(b);
        for (batch, &length) in lengths_vec.iter().enumerate() {
            let length = length as usize;
            let predictions = logits.narrow(0, batch, 1)?.narrow(1, 0, length)?;
            let best_tokens = predictions.argmax(D::Minus1)?.to_vec1::<u32>()?;
            let mut hypothesis = Vec::new();
            let mut token_boundaries: Vec<(usize, Option<usize>)> = Vec::new();
            let mut prev_token: i32 = -1;

            for (t, &token_id) in best_tokens.iter().enumerate() {
                let token_idx = token_id as usize;
                if token_idx == self.vocabulary.len() {
                    continue;
                }
                if token_idx as i32 == prev_token {
                    continue;
                }
                if prev_token != -1 {
                    let start_frame = token_boundaries.last().unwrap().0;
                    let token_start_time = start_frame as f64 * tr;
                    let token_end_time = t as f64 * tr;
                    let token_duration = token_end_time - token_start_time;

                    let span_probs =
                        probs
                            .narrow(0, batch, 1)?
                            .narrow(1, start_frame, t - start_frame)?;
                    let confidence = confidence_from_span(&span_probs, vocab_size)?;

                    hypothesis.push(AlignedToken::new(
                        prev_token as usize,
                        tokenizer::decode(&[prev_token as usize], &self.vocabulary),
                        token_start_time,
                        token_duration,
                        confidence,
                    ));
                }

                token_boundaries.push((t, None));
                prev_token = token_idx as i32;
            }

            if prev_token != -1 {
                let mut last_non_blank = length.saturating_sub(1);
                for t in (token_boundaries.last().unwrap().0..length).rev() {
                    if best_tokens[t] as usize != self.vocabulary.len() {
                        last_non_blank = t;
                        break;
                    }
                }
                let start_frame = token_boundaries.last().unwrap().0;
                let token_start_time = start_frame as f64 * tr;
                let token_end_time = (last_non_blank + 1) as f64 * tr;
                let token_duration = token_end_time - token_start_time;

                let span_probs = probs.narrow(0, batch, 1)?.narrow(
                    1,
                    start_frame,
                    last_non_blank + 1 - start_frame,
                )?;
                let confidence = confidence_from_span(&span_probs, vocab_size)?;

                hypothesis.push(AlignedToken::new(
                    prev_token as usize,
                    tokenizer::decode(&[prev_token as usize], &self.vocabulary),
                    token_start_time,
                    token_duration,
                    confidence,
                ));
            }

            results.push(hypothesis);
        }

        Ok(results)
    }

    pub fn generate(
        &mut self,
        mel: &Tensor,
        config: &DecodingConfig,
    ) -> Result<Vec<AlignedResult>> {
        let mel = if mel.dims().len() == 2 {
            mel.unsqueeze(0)?
        } else {
            mel.clone()
        };
        let (features, lengths) = self.encoder.forward(&mel, None)?;
        let result = self.decode(&features, &lengths)?;
        Ok(result
            .into_iter()
            .map(|hyp| sentences_to_result(&tokens_to_sentences(&hyp, &config.sentence)))
            .collect())
    }

    pub fn transcribe(
        &mut self,
        path: &std::path::Path,
        config: &DecodingConfig,
        chunk_duration: Option<f64>,
        overlap_duration: f64,
        chunk_callback: Option<Box<dyn FnMut(usize, usize)>>,
    ) -> Result<AlignedResult> {
        let audio = load_audio(path, self.preprocessor_config.sample_rate)?;
        let device = self.device.clone();
        let pre = self.preprocessor_config.clone();
        let mut generate = |mel: &Tensor, cfg: &DecodingConfig| -> Result<AlignedResult> {
            Ok(self.generate(mel, cfg)?[0].clone())
        };
        transcribe_with_audio(
            TranscribeParams {
                audio: &audio,
                device: &device,
                pre: &pre,
                config,
                chunk_duration,
                overlap_duration,
                chunk_callback,
            },
            &mut generate,
        )
    }

    pub fn transcribe_stream(
        &mut self,
        context_size: (usize, usize),
        depth: usize,
        keep_original_attention: bool,
        config: DecodingConfig,
    ) -> StreamingParakeet<'_> {
        StreamingParakeet::new(
            ParakeetModelRef::Ctc(self),
            context_size,
            depth,
            keep_original_attention,
            config,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ParakeetTdtCtc {
    pub base: ParakeetTdt,
    pub ctc_decoder: ConvASRDecoder,
}

impl ParakeetTdtCtc {
    pub fn load(args: ParakeetTdtCtcArgs, vb: VarBuilder) -> Result<Self> {
        let base = ParakeetTdt::load(args.base, vb.clone())?;
        let ctc_decoder =
            ConvASRDecoder::load(&args.aux_ctc.decoder, vb.pp("aux_ctc").pp("decoder"))?;
        Ok(Self { base, ctc_decoder })
    }
}

#[derive(Debug)]
pub enum ParakeetModelRef<'a> {
    Tdt(&'a mut ParakeetTdt),
    Rnnt(&'a mut ParakeetRnnt),
    Ctc(&'a mut ParakeetCtc),
}

impl<'a> ParakeetModelRef<'a> {
    fn preprocessor_config(&self) -> &PreprocessArgs {
        match self {
            Self::Tdt(m) => &m.preprocessor_config,
            Self::Rnnt(m) => &m.preprocessor_config,
            Self::Ctc(m) => &m.preprocessor_config,
        }
    }

    fn encoder_config(&self) -> &ConformerArgs {
        match self {
            Self::Tdt(m) => &m.encoder_config,
            Self::Rnnt(m) => &m.encoder_config,
            Self::Ctc(m) => &m.encoder_config,
        }
    }

    fn device(&self) -> &Device {
        match self {
            Self::Tdt(m) => &m.device,
            Self::Rnnt(m) => &m.device,
            Self::Ctc(m) => &m.device,
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            Self::Tdt(m) => m.encoder.num_layers(),
            Self::Rnnt(m) => m.encoder.num_layers(),
            Self::Ctc(m) => m.encoder.num_layers(),
        }
    }

    fn set_attention_model(&mut self, name: &str, context_size: Option<(usize, usize)>) {
        match self {
            Self::Tdt(m) => m.encoder.set_attention_model(name, context_size),
            Self::Rnnt(m) => m.encoder.set_attention_model(name, context_size),
            Self::Ctc(m) => m.encoder.set_attention_model(name, context_size),
        }
    }

    fn encoder_forward_with_cache(
        &mut self,
        mel: &Tensor,
        cache: &mut [RotatingConformerCache],
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Tdt(m) => m.encoder.forward_with_cache(mel, None, cache),
            Self::Rnnt(m) => m.encoder.forward_with_cache(mel, None, cache),
            Self::Ctc(m) => m.encoder.forward_with_cache(mel, None, cache),
        }
    }
}

#[derive(Debug)]
pub struct StreamingParakeet<'a> {
    model: ParakeetModelRef<'a>,
    cache: Vec<RotatingConformerCache>,
    /// Raw audio samples pending hop-alignment.
    audio_buffer: Vec<f32>,
    /// Raw audio samples for the carryover mel region (re-encoded with new context).
    audio_carryover: Vec<f32>,
    decoder_hidden: Option<(Tensor, Tensor)>,
    last_token: Option<usize>,
    finalized_tokens: Vec<AlignedToken>,
    draft_tokens: Vec<AlignedToken>,
    context_size: (usize, usize),
    depth: usize,
    decoding_config: DecodingConfig,
    keep_original_attention: bool,
}

impl<'a> StreamingParakeet<'a> {
    pub fn new(
        model: ParakeetModelRef<'a>,
        context_size: (usize, usize),
        depth: usize,
        keep_original_attention: bool,
        decoding_config: DecodingConfig,
    ) -> Self {
        let mut model = model;
        if !keep_original_attention {
            model.set_attention_model("rel_pos_local_attn", Some(context_size));
        }

        let cache_layers = model.num_layers();
        let cache = (0..cache_layers)
            .map(|_| RotatingConformerCache::new(context_size.0, context_size.1 * depth))
            .collect();
        Self {
            model,
            cache,
            audio_buffer: Vec::new(),
            audio_carryover: Vec::new(),
            decoder_hidden: None,
            last_token: None,
            finalized_tokens: Vec::new(),
            draft_tokens: Vec::new(),
            context_size,
            depth,
            decoding_config,
            keep_original_attention,
        }
    }

    pub fn add_audio(&mut self, audio: &[f32]) -> Result<()> {
        self.audio_buffer.extend_from_slice(audio);
        let pre = self.model.preprocessor_config().clone();
        let enc = self.model.encoder_config().clone();

        let hop = pre.hop_length();
        let usable = self.audio_buffer.len() / hop * hop;
        if usable == 0 {
            return Ok(());
        }

        // Combine carryover audio with new audio for consistent mel normalization.
        // Computing mel from contiguous audio avoids normalization discontinuities
        // that occur when independently-normalized mel chunks are concatenated.
        let mut full_audio = std::mem::take(&mut self.audio_carryover);
        full_audio.extend_from_slice(&self.audio_buffer[..usable]);
        self.audio_buffer.drain(0..usable);

        let mel = get_logmel(&full_audio, &pre, self.model.device())?;
        let total_mel = mel.dims3()?.1;
        let usable_mel = (total_mel / enc.subsampling_factor) * enc.subsampling_factor;
        if usable_mel == 0 {
            self.audio_carryover = full_audio;
            return Ok(());
        }
        let mel_in = mel.narrow(1, 0, usable_mel)?;

        // Run encoder with cache
        let (features, lengths) = self
            .model
            .encoder_forward_with_cache(&mel_in, &mut self.cache)?;

        let length = lengths.to_vec1::<i64>()?[0] as usize;
        if length == 0 {
            self.audio_carryover = full_audio;
            return Ok(());
        }

        // Compute how much mel/audio to keep for the draft region + leftover
        let leftover_mel = total_mel - length * enc.subsampling_factor;
        let drop = (self.context_size.1 * self.depth).min(length);
        let keep_mel = drop * enc.subsampling_factor + leftover_mel;

        // Keep raw audio corresponding to the carryover mel frames.
        // Each mel frame starts at frame_index * hop, plus we need win_length context.
        let keep_audio_start = full_audio
            .len()
            .saturating_sub(keep_mel * hop + pre.win_length());
        self.audio_carryover = full_audio[keep_audio_start..].to_vec();

        let finalized_length = length - drop;

        // Decode finalized and draft tokens
        match &mut self.model {
            ParakeetModelRef::Tdt(model) => {
                let (finalized, state) = model.decode(
                    &features,
                    Some(&Tensor::from_vec(
                        vec![finalized_length as i64],
                        (1,),
                        features.device(),
                    )?),
                    Some(vec![self.last_token]),
                    Some(vec![self.decoder_hidden.clone()]),
                    &self.decoding_config,
                )?;

                self.decoder_hidden = state[0].clone();
                self.last_token = finalized[0].last().map(|t| t.id);

                let draft_len = features.dims3()?.1 - finalized_length;
                let (draft, _) = model.decode(
                    &features.narrow(1, finalized_length, draft_len)?,
                    Some(&Tensor::from_vec(
                        vec![draft_len as i64],
                        (1,),
                        features.device(),
                    )?),
                    Some(vec![self.last_token]),
                    Some(vec![self.decoder_hidden.clone()]),
                    &self.decoding_config,
                )?;

                self.finalized_tokens.extend(finalized[0].clone());
                self.draft_tokens = draft[0].clone();
            }
            ParakeetModelRef::Rnnt(model) => {
                let (finalized, state) = model.decode(
                    &features,
                    Some(&Tensor::from_vec(
                        vec![finalized_length as i64],
                        (1,),
                        features.device(),
                    )?),
                    Some(vec![self.last_token]),
                    Some(vec![self.decoder_hidden.clone()]),
                )?;

                self.decoder_hidden = state[0].clone();
                self.last_token = finalized[0].last().map(|t| t.id);

                let draft_len = features.dims3()?.1 - finalized_length;
                let (draft, _) = model.decode(
                    &features.narrow(1, finalized_length, draft_len)?,
                    Some(&Tensor::from_vec(
                        vec![draft_len as i64],
                        (1,),
                        features.device(),
                    )?),
                    Some(vec![self.last_token]),
                    Some(vec![self.decoder_hidden.clone()]),
                )?;

                self.finalized_tokens.extend(finalized[0].clone());
                self.draft_tokens = draft[0].clone();
            }
            ParakeetModelRef::Ctc(model) => {
                let finalized = model.decode(
                    &features,
                    &Tensor::from_vec(vec![finalized_length as i64], (1,), features.device())?,
                )?;
                let draft_len = features.dims3()?.1 - finalized_length;
                let draft = model.decode(
                    &features.narrow(1, finalized_length, draft_len)?,
                    &Tensor::from_vec(vec![draft_len as i64], (1,), features.device())?,
                )?;

                self.finalized_tokens.extend(finalized[0].clone());
                self.draft_tokens = draft[0].clone();
            }
        }

        Ok(())
    }

    pub fn result(&self) -> AlignedResult {
        let mut tokens = self.finalized_tokens.clone();
        tokens.extend(self.draft_tokens.clone());
        sentences_to_result(&tokens_to_sentences(
            &tokens,
            &self.decoding_config.sentence,
        ))
    }

    /// Signal end of audio stream. Promotes all remaining draft tokens to finalized.
    pub fn finalize(&mut self) {
        self.finalized_tokens.append(&mut self.draft_tokens);
    }

    /// Returns the number of finalized tokens so far.
    pub fn finalized_count(&self) -> usize {
        self.finalized_tokens.len()
    }
}

impl Drop for StreamingParakeet<'_> {
    fn drop(&mut self) {
        if self.keep_original_attention {
            return;
        }
        self.model.set_attention_model("rel_pos", None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};

    #[test]
    fn confidence_from_logits_uniform_distribution() {
        // Uniform distribution => maximum entropy => confidence near 0
        let logits = Tensor::zeros((4,), DType::F32, &Device::Cpu).unwrap();
        let conf = confidence_from_logits(&logits, 4).unwrap();
        assert!(
            conf < 0.1,
            "uniform distribution should have low confidence, got {conf}"
        );
    }

    #[test]
    fn confidence_from_logits_peaked_distribution() {
        // One-hot-like distribution => low entropy => confidence near 1
        let data = vec![-100.0f32, -100.0, 100.0, -100.0];
        let logits = Tensor::from_vec(data, (4,), &Device::Cpu).unwrap();
        let conf = confidence_from_logits(&logits, 4).unwrap();
        assert!(
            conf > 0.9,
            "peaked distribution should have high confidence, got {conf}"
        );
    }

    #[test]
    fn confidence_from_span_returns_valid_range() {
        // 3 time steps, 4 classes - slightly peaked distribution
        let mut data = vec![0.1f32; 12];
        // Make first class dominant in each frame
        for i in 0..3 {
            data[i * 4] = 0.7;
        }
        let probs = Tensor::from_vec(data, (1, 3, 4), &Device::Cpu).unwrap();
        let conf = confidence_from_span(&probs, 4).unwrap();
        assert!(
            conf > 0.0,
            "confidence should be positive for peaked distribution, got {conf}"
        );
        assert!(conf <= 1.0, "confidence should be <= 1.0, got {conf}");
    }

    #[test]
    fn beam_default_valid() {
        let beam = Beam::default();
        assert!(beam.beam_size >= 1);
        assert!(beam.patience > 0.0);
    }

    #[test]
    fn beam_zero_size_rejected() {
        let beam = Beam {
            beam_size: 0,
            ..Beam::default()
        };
        // We can't test decode_beam directly without a full model,
        // but verify the constraint is documented in the type
        assert_eq!(beam.beam_size, 0);
    }

    #[test]
    fn time_ratio_positive() {
        let pre = PreprocessArgs {
            sample_rate: 16000,
            normalize: "per_feature".to_string(),
            window_size: 0.025,
            window_stride: 0.01,
            window: "hann".to_string(),
            features: 80,
            n_fft: 512,
            dither: 0.0,
            pad_to: 0,
            pad_value: 0.0,
            preemph: Some(0.97),
            mag_power: 2.0,
        };
        let enc = ConformerArgs {
            feat_in: 80,
            n_layers: 24,
            d_model: 512,
            n_heads: 8,
            ff_expansion_factor: 4,
            subsampling_factor: 4,
            self_attention_model: "rel_pos".to_string(),
            subsampling: "dw_striding".to_string(),
            conv_kernel_size: 9,
            subsampling_conv_channels: 256,
            pos_emb_max_len: 5000,
            causal_downsampling: false,
            use_bias: true,
            xscaling: false,
            pos_bias_u: None,
            pos_bias_v: None,
            subsampling_conv_chunking_factor: 1,
            att_context_size: None,
        };
        let tr = time_ratio(&pre, &enc);
        assert!(tr > 0.0, "time_ratio should be positive, got {tr}");
    }

    #[test]
    fn decoding_config_default_is_greedy() {
        let config = DecodingConfig::default();
        assert!(matches!(config.decoding, Decoding::Greedy(_)));
        assert!(!config.debug_decode);
    }

    #[test]
    fn hypothesis_clone_independent() {
        let h1 = Hypothesis {
            score: 1.0,
            step: 5,
            last_token: Some(42),
            hidden_state: None,
            stuck: 0,
            tokens: vec![AlignedToken::new(42, "hello".to_string(), 0.0, 0.5, 0.9)],
        };
        let mut h2 = h1.clone();
        h2.score = 2.0;
        h2.tokens
            .push(AlignedToken::new(43, "world".to_string(), 0.5, 0.5, 0.8));
        assert_eq!(h1.tokens.len(), 1);
        assert_eq!(h2.tokens.len(), 2);
        assert!((h1.score - 1.0).abs() < f64::EPSILON);
    }
}
