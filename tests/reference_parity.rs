//! Golden-output parity tests against parakeet-mlx.
//!
//! Fixtures live in `tests/fixtures/reference/` and come from
//! `tests/fixtures/generate_reference.py` (parakeet-mlx 0.7.0, MLX 0.32.4 CPU,
//! float32). See `tests/fixtures/README.md` for how to regenerate them.
//!
//! Tests that need the model weights are `#[ignore]`d. Run them with:
//!
//! ```sh
//! cargo test --release --test reference_parity -- --ignored --skip known_bug --nocapture
//! ```
//!
//! `known_bug_*` tests reproduce reported bugs and fail until those are fixed.
//!
//! The first run downloads `mlx-community/parakeet-tdt-0.6b-v3` (about 2.5 GB)
//! at the revision the fixtures were generated from.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{HFClientSync, split_id};
use parakeet_candle::parakeet::{
    AlignedResult, AlignedToken, Beam, Decoding, DecodingConfig, ParakeetModel, ParakeetTdt,
    PreprocessArgs, from_config_value, get_logmel, load_audio,
};
use serde::Deserialize;

// Tolerances. Each one is justified next to the assertion that uses it.

/// Mel tolerances: see `logmel_matches_reference`.
const MEL_MAX_ABS: f32 = 4e-3;
const MEL_MEAN_ABS: f32 = 5e-4;

/// Encoder tolerances: see `encoder_matches_reference`.
const ENCODER_MAX_ABS: f32 = 1e-5;
const ENCODER_REL_L2: f32 = 1e-4;

/// Confidence is `1 - entropy / ln(V)` over the joint softmax, so it moves
/// with the encoder output. Measured differences are below 4e-6; 1e-4 leaves
/// headroom while still catching a changed joint or softmax.
const CONFIDENCE_ABS: f64 = 1e-4;

/// Timestamps are integer multiples of one encoder frame (80 ms). Identical
/// decoding paths give bit-identical frame indices; 1e-6 only absorbs the
/// f32/f64 formatting difference.
const TIME_ABS: f64 = 1e-6;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference")
}

#[derive(Debug, Deserialize)]
struct Reference {
    model_id: String,
    model_revision: String,
    preprocessor: PreprocessArgs,
    audio: AudioRef,
    mel: MelRef,
    encoder: EncoderRef,
    greedy: TranscriptRef,
    greedy_full_precision_mel: TranscriptRef,
    beam: BeamRef,
    streaming: Vec<StreamingRef>,
    parakeet_mlx_l1_magnitude: L1MagnitudeRef,
}

/// What stock parakeet-mlx (|re| + |im| STFT magnitude) produces. Recorded
/// for the report, not asserted.
#[derive(Debug, Deserialize)]
struct L1MagnitudeRef {
    mel_max_abs_diff: f32,
    greedy: TranscriptRef,
}

#[derive(Debug, Deserialize)]
struct AudioRef {
    file: String,
    sample_rate: usize,
    num_samples: usize,
    sum: f64,
    sum_sq: f64,
}

#[derive(Debug, Deserialize)]
struct MelRef {
    shape: Vec<usize>,
    f16_rounding_max_abs: f32,
}

#[derive(Debug, Deserialize)]
struct EncoderRef {
    shape: Vec<usize>,
    length: usize,
}

#[derive(Debug, Deserialize)]
struct TokenRef {
    id: usize,
    text: String,
    start: f64,
    duration: f64,
    confidence: f64,
}

#[derive(Debug, Deserialize)]
struct TranscriptRef {
    text: String,
    tokens: Vec<TokenRef>,
}

#[derive(Debug, Deserialize)]
struct BeamRef {
    beam_size: usize,
    #[serde(flatten)]
    transcript: TranscriptRef,
}

#[derive(Debug, Deserialize)]
struct StreamingRef {
    context_size: (usize, usize),
    depth: usize,
    chunk_seconds: f64,
    #[serde(flatten)]
    transcript: TranscriptRef,
}

fn reference() -> Reference {
    let path = fixture_dir().join("reference.json");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_reader(file).expect("parse reference.json")
}

fn tensors() -> HashMap<String, Tensor> {
    candle::safetensors::load(fixture_dir().join("tensors.safetensors"), &Device::Cpu)
        .expect("load tensors.safetensors")
}

fn tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Tensor {
    tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing fixture tensor {name}"))
        .to_dtype(DType::F32)
        .unwrap()
}

fn clip(reference: &Reference) -> Vec<f32> {
    load_audio(
        &fixture_dir().join(&reference.audio.file),
        reference.audio.sample_rate,
    )
    .expect("decode fixture clip")
}

struct Diff {
    max_abs: f32,
    mean_abs: f32,
    rel_l2: f32,
}

fn diff(actual: &Tensor, expected: &Tensor) -> Diff {
    assert_eq!(actual.dims(), expected.dims(), "shape mismatch");
    let a = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let e = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut err_sq = 0f64;
    let mut ref_sq = 0f64;
    for (&x, &y) in a.iter().zip(&e) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        err_sq += (d as f64).powi(2);
        ref_sq += (y as f64).powi(2);
    }
    Diff {
        max_abs,
        mean_abs: (sum_abs / a.len() as f64) as f32,
        rel_l2: (err_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt() as f32,
    }
}

/// Word-level edit distance of `hyp` from `reference`.
fn word_edits(reference: &str, hyp: &str) -> usize {
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hyp.split_whitespace().collect();
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hw) in h.iter().enumerate() {
            let sub = prev[j] + usize::from(rw != hw);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()]
}

// ---------------------------------------------------------------------------
// Default tests: no model weights needed.
// ---------------------------------------------------------------------------

/// The FLAC is 16 kHz mono s16, so decoding needs no resampling and both
/// sides compute `sample / 32768` exactly. Any difference means the decoder
/// or the int-to-float scaling changed.
#[test]
fn clip_decodes_to_reference_samples() {
    let reference = reference();
    let tensors = tensors();
    let audio = clip(&reference);
    assert_eq!(audio.len(), reference.audio.num_samples);

    let head = tensor(&tensors, "audio_head").to_vec1::<f32>().unwrap();
    assert_eq!(&audio[..head.len()], head.as_slice());

    let sum: f64 = audio.iter().map(|&x| x as f64).sum();
    let sum_sq: f64 = audio.iter().map(|&x| (x as f64).powi(2)).sum();
    assert!((sum - reference.audio.sum).abs() < 1e-9, "sum {sum}");
    assert!(
        (sum_sq - reference.audio.sum_sq).abs() < 1e-9,
        "sum_sq {sum_sq}"
    );
}

/// Compares `get_logmel` with the reference front end (parakeet-mlx's
/// `get_logmel` with the true STFT magnitude) on the same samples.
///
/// The fixture mel is stored as f16. For normalised values in [4, 8) half an
/// f16 ulp is 1.95e-3, which is the measured max_abs (mean_abs 1.3e-4). The
/// bounds allow about 2x that for f32 STFT and log noise. A filterbank,
/// window, or magnitude change moves max_abs to 1e-1 or more.
#[test]
fn logmel_matches_reference() {
    let reference = reference();
    let tensors = tensors();
    let audio = clip(&reference);
    let mel = get_logmel(&audio, &reference.preprocessor, &Device::Cpu).unwrap();
    assert_eq!(mel.dims(), reference.mel.shape.as_slice());

    let expected = tensor(&tensors, "mel_f16");
    let d = diff(&mel, &expected);
    eprintln!(
        "mel: max_abs={:.3e} mean_abs={:.3e} rel_l2={:.3e} (f16 storage rounding up to {:.3e})",
        d.max_abs, d.mean_abs, d.rel_l2, reference.mel.f16_rounding_max_abs
    );
    assert!(d.max_abs <= MEL_MAX_ABS, "mel max_abs {}", d.max_abs);
    assert!(d.mean_abs <= MEL_MEAN_ABS, "mel mean_abs {}", d.mean_abs);
}

// ---------------------------------------------------------------------------
// Model tests: `#[ignore]`d, need the weights from the Hugging Face Hub.
// ---------------------------------------------------------------------------

fn load_model(reference: &Reference) -> ParakeetModel {
    let api = HFClientSync::new().unwrap();
    let (owner, name) = split_id(&reference.model_id);
    let repo = api.model(owner, name);
    let fetch = |filename: &str| -> PathBuf {
        repo.download_file()
            .filename(filename)
            .revision(reference.model_revision.clone())
            .send()
            .unwrap_or_else(|e| panic!("download {filename}: {e}"))
    };
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(fetch("config.json")).unwrap()).unwrap();
    let weights = fetch("model.safetensors");
    // SAFETY: the Hub cache file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &Device::Cpu) }
        .unwrap();
    from_config_value(config, vb).unwrap()
}

fn tdt(model: &mut ParakeetModel) -> &mut ParakeetTdt {
    match model {
        ParakeetModel::Tdt(m) => m,
        other => panic!("fixtures are for a TDT model, got {other:?}"),
    }
}

/// Compares token ids and text exactly, and timing and confidence within the
/// tolerances above.
fn assert_transcript(label: &str, actual: &AlignedResult, expected: &TranscriptRef) {
    let tokens: Vec<AlignedToken> = actual.tokens();
    eprintln!("{label}: {:?}", actual.text);
    assert_eq!(actual.text, expected.text, "{label}: transcript text");
    assert_eq!(tokens.len(), expected.tokens.len(), "{label}: token count");
    let mut worst_conf = 0f64;
    for (i, (a, e)) in tokens.iter().zip(&expected.tokens).enumerate() {
        assert_eq!(
            (a.id, a.text.as_str()),
            (e.id, e.text.as_str()),
            "{label}: token {i}"
        );
        assert!(
            (a.start - e.start).abs() <= TIME_ABS && (a.duration - e.duration).abs() <= TIME_ABS,
            "{label}: token {i} {:?} at {}+{} expected {}+{}",
            a.text,
            a.start,
            a.duration,
            e.start,
            e.duration
        );
        // f64::max ignores NaN, so check finiteness before aggregating.
        assert!(
            a.confidence.is_finite(),
            "{label}: token {i} confidence {}",
            a.confidence
        );
        worst_conf = worst_conf.max((a.confidence - e.confidence).abs());
    }
    eprintln!("{label}: max confidence diff {worst_conf:.3e}");
    assert!(
        worst_conf <= CONFIDENCE_ABS,
        "{label}: confidence diff {worst_conf}"
    );
}

/// Runs the Rust encoder on the reference mel, so mel differences cannot leak
/// in, and compares eight full 1024-d frames plus per-frame and per-channel
/// statistics over all 88 frames.
///
/// Encoder outputs have abs max about 0.2 and std about 0.026. Through 24
/// conformer layers in f32 with a different matmul and reduction order (MLX
/// vs gemm), the measured error is max_abs 3e-7 and rel_l2 up to 7e-6 (on
/// x86-64, 2026-11). The bounds leave about 30x headroom for other BLAS
/// backends and CPUs. A wrong weight, layout, or op gives errors of 1e-2 or
/// more.
#[test]
#[ignore = "needs model weights from the Hugging Face Hub"]
fn encoder_matches_reference() {
    let reference = reference();
    let tensors = tensors();
    let mut model = load_model(&reference);
    let m = tdt(&mut model);

    let mel = tensor(&tensors, "mel_f16");
    let (features, lengths) = m.encoder.forward(&mel, None).unwrap();
    assert_eq!(features.dims(), reference.encoder.shape.as_slice());
    assert_eq!(
        lengths.to_vec1::<i64>().unwrap(),
        vec![reference.encoder.length as i64]
    );
    let features = features.i(0).unwrap();

    let frames = tensors["encoder_row_frames"].to_vec1::<i32>().unwrap();
    let frames: Vec<u32> = frames.iter().map(|&f| f as u32).collect();
    let index = Tensor::new(frames.as_slice(), &Device::Cpu).unwrap();
    let rows = features.index_select(&index, 0).unwrap();

    let checks = [
        ("rows", rows, tensor(&tensors, "encoder_rows")),
        (
            "frame_mean",
            features.mean(1).unwrap(),
            tensor(&tensors, "encoder_frame_mean"),
        ),
        (
            "frame_std",
            features
                .broadcast_sub(&features.mean_keepdim(1).unwrap())
                .unwrap()
                .sqr()
                .unwrap()
                .mean(1)
                .unwrap()
                .sqrt()
                .unwrap(),
            tensor(&tensors, "encoder_frame_std"),
        ),
        (
            "channel_mean",
            features.mean(0).unwrap(),
            tensor(&tensors, "encoder_channel_mean"),
        ),
    ];
    for (name, actual, expected) in checks {
        let d = diff(&actual, &expected);
        eprintln!(
            "encoder {name}: max_abs={:.3e} mean_abs={:.3e} rel_l2={:.3e}",
            d.max_abs, d.mean_abs, d.rel_l2
        );
        assert!(
            d.max_abs <= ENCODER_MAX_ABS,
            "encoder {name} max_abs {}",
            d.max_abs
        );
        assert!(
            d.rel_l2 <= ENCODER_REL_L2,
            "encoder {name} rel_l2 {}",
            d.rel_l2
        );
    }
}

/// Greedy TDT decoding of the reference mel: the full model minus the
/// front end.
#[test]
#[ignore = "needs model weights from the Hugging Face Hub"]
fn greedy_matches_reference() {
    let reference = reference();
    let tensors = tensors();
    let mut model = load_model(&reference);
    let mel = tensor(&tensors, "mel_f16");
    let result = model.generate(&mel, &DecodingConfig::default()).unwrap();
    assert_transcript("greedy", &result[0], &reference.greedy);
}

/// Beam-search TDT decoding of the reference mel with parakeet-mlx's default
/// beam parameters.
#[test]
#[ignore = "needs model weights from the Hugging Face Hub"]
fn beam_matches_reference() {
    let reference = reference();
    let tensors = tensors();
    let mut model = load_model(&reference);
    let mel = tensor(&tensors, "mel_f16");
    let config = DecodingConfig {
        decoding: Decoding::Beam(Beam {
            beam_size: reference.beam.beam_size,
            ..Beam::default()
        }),
        ..DecodingConfig::default()
    };
    let result = model.generate(&mel, &config).unwrap();
    assert_transcript("beam", &result[0], &reference.beam.transcript);
}

/// Audio file to transcript through the Rust front end. Token ids and timing
/// must match the reference run on its own full-precision mel.
///
/// The reference mel uses the true STFT magnitude, as NeMo does. Stock
/// parakeet-mlx uses |re| + |im|; its transcript is printed for comparison.
#[test]
#[ignore = "needs model weights from the Hugging Face Hub"]
fn transcribe_file_matches_reference() {
    let reference = reference();
    let mut model = load_model(&reference);
    let result = model
        .transcribe(
            &fixture_dir().join(&reference.audio.file),
            &DecodingConfig::default(),
            None,
            15.0,
            None,
        )
        .unwrap();
    let stock = &reference.parakeet_mlx_l1_magnitude;
    eprintln!(
        "stock parakeet-mlx front end (mel differs by up to {:.2}): {:?}",
        stock.mel_max_abs_diff, stock.greedy.text
    );
    assert_transcript("transcribe", &result, &reference.greedy_full_precision_mel);
}

/// Streams the clip through `StreamingParakeet` with the settings of one
/// recorded parakeet-mlx run, in chunks of `chunk_seconds`.
fn stream_clip(
    model: &mut ParakeetModel,
    audio: &[f32],
    sample_rate: usize,
    s: &StreamingRef,
) -> candle::Result<AlignedResult> {
    let chunk = (s.chunk_seconds * sample_rate as f64) as usize;
    let mut stream =
        model.transcribe_stream(s.context_size, s.depth, false, DecodingConfig::default());
    for samples in audio.chunks(chunk) {
        stream.add_audio(samples)?;
    }
    stream.finalize();
    Ok(stream.result())
}

fn streaming_config(reference: &Reference, context_size: (usize, usize)) -> &StreamingRef {
    reference
        .streaming
        .iter()
        .find(|s| s.context_size == context_size)
        .unwrap_or_else(|| panic!("no streaming fixture for context {context_size:?}"))
}

/// Streams the clip for one configuration, prints the result next to the
/// offline and parakeet-mlx streaming transcripts, and returns an error
/// message if it fails or has more word edits than the budget allows.
///
/// Budget: at most one word edit more than parakeet-mlx streaming makes with
/// the same settings, measured against the offline transcript. Streaming is
/// not expected to equal offline decoding: local attention with `depth = 1`
/// reproduces only the first layer exactly, and each chunk's mel is
/// normalised over a partial window, differently in the two implementations.
/// One extra edit absorbs that; a broken cache or decoder state costs many.
fn check_streaming(
    model: &mut ParakeetModel,
    reference: &Reference,
    audio: &[f32],
    s: &StreamingRef,
) -> Option<String> {
    let offline = &reference.greedy_full_precision_mel.text;
    let words = offline.split_whitespace().count();
    let label = format!(
        "context={:?} depth={} chunk={}s",
        s.context_size, s.depth, s.chunk_seconds
    );
    let mlx_edits = word_edits(offline, &s.transcript.text);
    eprintln!("{label}");
    eprintln!("  offline    : {offline:?}");
    eprintln!(
        "  mlx stream : {:?} ({mlx_edits}/{words} word edits)",
        s.transcript.text
    );

    let result = match stream_clip(model, audio, reference.audio.sample_rate, s) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("  rust stream: error: {e}");
            return Some(format!("{label}: add_audio failed: {e}"));
        }
    };
    let rust_edits = word_edits(offline, &result.text);
    eprintln!(
        "  rust stream: {:?} ({rust_edits}/{words} word edits)",
        result.text
    );
    eprintln!(
        "  rust tokens: {:?}",
        result
            .iter_tokens()
            .map(|t| (t.text.as_str(), t.start, t.duration))
            .collect::<Vec<_>>()
    );
    (rust_edits > mlx_edits + 1)
        .then(|| format!("{label}: Rust {rust_edits} word edits vs parakeet-mlx {mlx_edits}"))
}

/// Streams the clip in 0.5 s chunks with contexts whose left side (in
/// encoder frames) covers the whole 88-frame clip, so the rotating KV cache
/// never wraps:
///
/// - (256, 256), the parakeet-mlx default. Every frame stays in the draft
///   region, so each chunk re-encodes the clip so far.
/// - (128, 16). Frames older than 16 are committed, which exercises the
///   KV and conv caches across chunks.
#[test]
#[ignore = "needs model weights from the Hugging Face Hub"]
fn streaming_close_to_offline() {
    let reference = reference();
    let audio = clip(&reference);
    let mut model = load_model(&reference);
    let failures: Vec<String> = [(256, 256), (128, 16)]
        .into_iter()
        .filter_map(|context| {
            check_streaming(
                &mut model,
                &reference,
                &audio,
                streaming_config(&reference, context),
            )
        })
        .collect();

    // Dropping the stream restores full attention, so offline decoding after
    // streaming must be unaffected.
    let mel = get_logmel(&audio, model.preprocessor_config(), &Device::Cpu).unwrap();
    let after = model.generate(&mel, &DecodingConfig::default()).unwrap();
    assert_eq!(
        after[0].text, reference.greedy_full_precision_mel.text,
        "offline transcript after streaming"
    );

    assert!(
        failures.is_empty(),
        "streaming failures:\n{}",
        failures.join("\n")
    );
}

/// Known bug, expected to fail until fixed: once more frames than the left
/// context (64) have been cached, `add_audio` fails with a shape mismatch.
/// The local-attention mask is sized from `cache.offset()`, which counts every
/// frame ever cached, while `RotatingConformerCache` keeps at most `capacity`
/// frames. The documented command skips this test; run it on its own with:
///
/// ```sh
/// cargo test --release --test reference_parity -- --ignored known_bug
/// ```
#[test]
#[ignore = "known bug: streaming fails once the rotating KV cache wraps"]
fn known_bug_streaming_cache_wrap() {
    let reference = reference();
    let audio = clip(&reference);
    let mut model = load_model(&reference);
    let s = streaming_config(&reference, (64, 16));
    if let Some(failure) = check_streaming(&mut model, &reference, &audio, s) {
        panic!("{failure}");
    }
}
