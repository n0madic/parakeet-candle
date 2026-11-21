#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::{HFClientSync, HFRepositorySync, RepoTypeModel, split_id};
use parakeet_candle::parakeet::{
    AlignedResult, AlignedSentence, AlignedToken, AudioReader, Beam, Decoding, DecodingConfig,
    Greedy, ParakeetModel, from_config_value, get_logmel, load_audio,
};

use candle::utils::{cuda_is_available, metal_is_available};
use candle::{DType, Device, IndexOp};
use candle_nn::VarBuilder;

#[derive(Clone, Debug, Default, clap::ValueEnum)]
enum OutputFormat {
    /// Plain text (default).
    #[default]
    Text,
    /// Machine-readable JSON.
    Json,
    /// SRT subtitles. Implies timestamps.
    Srt,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Parakeet ASR on Candle", long_about = None)]
struct Args {
    /// Run on CPU rather than GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Hugging Face model id.
    #[arg(long, default_value = "mlx-community/parakeet-tdt-0.6b-v3")]
    model_id: String,

    /// Input audio file path.
    #[arg(long)]
    input: PathBuf,

    /// Chunk duration in seconds for long audio.
    #[arg(long)]
    chunk_duration: Option<f64>,

    /// Overlap duration in seconds when chunking.
    #[arg(long, default_value_t = 15.0)]
    overlap_duration: f64,

    /// Beam size for TDT decoding.
    #[arg(long)]
    beam_size: Option<usize>,

    /// Print decoder internals and token alignment debug data.
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Print sentence timestamps after the transcript.
    #[arg(long, default_value_t = false)]
    timestamps: bool,

    /// Output format: text (default), json, srt.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    /// Use streaming mode for large file transcription (progressive output, lower memory).
    #[arg(long, default_value_t = false, conflicts_with = "chunk_duration")]
    stream: bool,

    /// Audio chunk duration in seconds per streaming iteration.
    #[arg(long, default_value_t = 30.0)]
    stream_chunk_secs: f64,
}

#[derive(serde::Serialize)]
struct JsonOutput<'a> {
    text: &'a str,
    model_id: &'a str,
    input: String,
    device: String,
    sentences: Option<&'a [AlignedSentence]>,
    tokens: Option<Vec<AlignedToken>>,
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        eprintln!("Running on CPU. Build with `--features metal` to enable Apple GPU.");

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        eprintln!("Running on CPU. Build with `--features cuda` to enable GPU.");

        Ok(Device::Cpu)
    }
}

fn hub_file(repo: &HFRepositorySync<RepoTypeModel>, filename: &str) -> hf_hub::HFResult<PathBuf> {
    repo.download_file()
        .filename(filename)
        .local_files_only(true)
        .send()
        .or_else(|_| repo.download_file().filename(filename).send())
}

fn hub_load_safetensors(
    repo: &HFRepositorySync<RepoTypeModel>,
    json_path: &Path,
) -> Result<Vec<PathBuf>> {
    let reader = File::open(json_path)?;
    let json: serde_json::Value = serde_json::from_reader(reader).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => anyhow::bail!("no weight map in {json_path:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => anyhow::bail!("weight map in {json_path:?} is not an object"),
    };

    let files = weight_map
        .values()
        .filter_map(|value| value.as_str())
        .map(ToOwned::to_owned)
        .collect::<HashSet<_>>();

    files
        .into_iter()
        .map(|file| hub_file(repo, &file).with_context(|| format!("loading weight shard {file}")))
        .collect()
}

use parakeet_candle::parakeet::PreprocessArgs;

fn debug_encoder(
    input: &Path,
    pre: &PreprocessArgs,
    device: &Device,
    encoder: &mut parakeet_candle::parakeet::conformer::Conformer,
) -> Result<(candle::Tensor, candle::Tensor)> {
    let audio = load_audio(input, pre.sample_rate)?;
    let audio_secs = audio.len() as f64 / pre.sample_rate as f64;
    eprintln!(
        "debug: audio_len={} samples secs={audio_secs:.2} sr={}",
        audio.len(),
        pre.sample_rate
    );
    let mel = get_logmel(&audio, pre, device)?;
    let (mb, mt, mf) = mel.dims3()?;
    eprintln!("debug: mel dims=({mb},{mt},{mf})");
    let (features, lengths) = encoder.forward(&mel, None)?;
    let (fb, ft, ff) = features.dims3()?;
    let lengths_vec = lengths.to_vec1::<i64>()?;
    eprintln!("debug: features dims=({fb},{ft},{ff}) lengths={lengths_vec:?}");
    Ok((features, lengths))
}

fn print_debug(model: &mut ParakeetModel, input: &Path) -> Result<()> {
    match model {
        ParakeetModel::Tdt(m) => {
            let (features, _) =
                debug_encoder(input, &m.preprocessor_config, &m.device, &mut m.encoder)?;

            let vocab = &m.vocabulary;
            let blank_id = vocab.len();
            let (decoder_out, _) = m.decoder.forward(None, None)?;
            let enc_step = features.narrow(1, 0, 1)?;
            let joint_out = m.joint.forward(&enc_step, &decoder_out)?;
            let vocab_size = vocab.len() + 1;
            let token_logits = joint_out.i((0, 0, 0, 0..vocab_size))?;
            let duration_logits = joint_out.i((0, 0, 0, vocab_size..))?;

            let token_vals = token_logits.to_vec1::<f32>()?;
            let mut idxs: Vec<usize> = (0..token_vals.len()).collect();
            idxs.sort_by(|&a, &b| {
                token_vals[b]
                    .partial_cmp(&token_vals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            eprintln!("debug: top token logits (step0)");
            for (i, id) in idxs.iter().take(10).enumerate() {
                let text = if *id == blank_id {
                    "<blank>".to_string()
                } else {
                    vocab
                        .get(*id)
                        .cloned()
                        .unwrap_or_else(|| "<oob>".to_string())
                };
                eprintln!(
                    "debug: tok[{i}] id={} text={text:?} logit={:.4}",
                    id, token_vals[*id]
                );
            }

            let duration_vals = duration_logits.to_vec1::<f32>()?;
            eprintln!("debug: duration logits {duration_vals:?}");
        }
        ParakeetModel::Rnnt(m) => {
            debug_encoder(input, &m.preprocessor_config, &m.device, &mut m.encoder)?;
        }
        ParakeetModel::Ctc(m) => {
            debug_encoder(input, &m.preprocessor_config, &m.device, &mut m.encoder)?;
        }
        ParakeetModel::TdtCtc(m) => {
            debug_encoder(
                input,
                &m.base.preprocessor_config,
                &m.base.device,
                &mut m.base.encoder,
            )?;
        }
    }
    Ok(())
}

fn print_timestamps(result: &AlignedResult) {
    for sentence in &result.sentences {
        println!(
            "[{:.3} - {:.3}] {}",
            sentence.start,
            sentence.end,
            sentence.text.trim()
        );
    }
}

fn format_srt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let millis = total_ms % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

fn print_srt(result: &AlignedResult) {
    for (idx, sentence) in result.sentences.iter().enumerate() {
        println!("{}", idx + 1);
        println!(
            "{} --> {}",
            format_srt_timestamp(sentence.start),
            format_srt_timestamp(sentence.end)
        );
        println!("{}", sentence.text.trim());
        println!();
    }
}

fn print_output(args: &Args, selected_device: &Device, result: &AlignedResult) -> Result<()> {
    let include_timestamps = args.timestamps || matches!(args.format, OutputFormat::Srt);

    if matches!(args.format, OutputFormat::Json) {
        let payload = JsonOutput {
            text: &result.text,
            model_id: &args.model_id,
            input: args.input.display().to_string(),
            device: format!("{selected_device:?}"),
            sentences: include_timestamps.then_some(result.sentences.as_slice()),
            tokens: include_timestamps.then(|| result.tokens()),
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    if matches!(args.format, OutputFormat::Srt) {
        print_srt(result);
        return Ok(());
    }

    println!("{}", result.text);
    if include_timestamps {
        println!();
        print_timestamps(result);
    }

    Ok(())
}

macro_rules! debug_log {
    ($start:expr, $($arg:tt)*) => {{
        let elapsed = $start.elapsed();
        eprintln!(
            "[{:>6.3}s] {}",
            elapsed.as_secs_f64(),
            format!($($arg)*)
        );
    }};
}

fn transcribe_streaming(
    args: &Args,
    model: &mut ParakeetModel,
    decoding_config: &DecodingConfig,
    t0: Instant,
) -> Result<AlignedResult> {
    use parakeet_candle::parakeet::alignment::{
        merge_longest_common_subsequence, merge_longest_contiguous, sentences_to_result,
        tokens_to_sentences,
    };
    use std::io::Write;

    let sample_rate = model.sample_rate();
    let mut audio_reader =
        AudioReader::open(&args.input, sample_rate).map_err(|e| anyhow::anyhow!("{e}"))?;

    let chunk_duration = args.stream_chunk_secs.max(5.0);
    let overlap_duration = chunk_duration.min(15.0);
    let chunk_samples = (chunk_duration * sample_rate as f64) as usize;
    let overlap_samples = (overlap_duration * sample_rate as f64) as usize;

    let pre = model.preprocessor_config().clone();
    let device = model.device_ref().clone();

    let mut all_tokens: Vec<AlignedToken> = Vec::new();
    let mut audio_offset: usize = 0;
    let mut printed_tokens: usize = 0;
    let mut leftover: Vec<f32> = Vec::new();
    let progressive = matches!(args.format, OutputFormat::Text) && !args.timestamps;

    loop {
        let mut chunk_audio = std::mem::take(&mut leftover);
        let needed = chunk_samples.saturating_sub(chunk_audio.len());
        if needed > 0 {
            let new = audio_reader
                .read_chunk(needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            if new.is_empty() && chunk_audio.is_empty() {
                break;
            }
            chunk_audio.extend_from_slice(&new);
        }

        if chunk_audio.len() < pre.hop_length() {
            break;
        }

        let mel = get_logmel(&chunk_audio, &pre, &device).map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut chunk_result = model
            .generate(&mel, decoding_config)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .swap_remove(0);

        let chunk_offset_secs = audio_offset as f64 / sample_rate as f64;
        for sentence in chunk_result.sentences.iter_mut() {
            for token in sentence.tokens.iter_mut() {
                token.start += chunk_offset_secs;
                token.end = token.start + token.duration;
            }
        }

        if all_tokens.is_empty() {
            all_tokens = chunk_result.tokens();
        } else {
            let new_tokens = chunk_result.tokens();
            match merge_longest_contiguous(&all_tokens, &new_tokens, overlap_duration) {
                Ok(tokens) => all_tokens = tokens,
                Err(_) => {
                    all_tokens = merge_longest_common_subsequence(
                        &all_tokens,
                        &new_tokens,
                        overlap_duration,
                    );
                }
            }
        }

        // The next merge can revise tokens that start in the overlap.
        if progressive && all_tokens.len() > printed_tokens {
            let safe_boundary =
                audio_offset as f64 / sample_rate as f64 + chunk_duration - overlap_duration;
            let safe_count = all_tokens
                .iter()
                .position(|t| t.start >= safe_boundary)
                .unwrap_or(all_tokens.len());

            for token in &all_tokens[printed_tokens..safe_count] {
                print!("{}", token.text);
            }
            if safe_count > printed_tokens {
                let _ = std::io::stdout().flush();
            }
            printed_tokens = safe_count;
        }

        if args.debug {
            debug_log!(
                t0,
                "streamed {:.1}s, tokens={}",
                audio_reader.position_secs(),
                all_tokens.len()
            );
        }

        let is_last = chunk_audio.len() < chunk_samples;
        if is_last {
            break;
        }

        let advance = chunk_samples.saturating_sub(overlap_samples);
        audio_offset += advance;
        if advance < chunk_audio.len() {
            leftover = chunk_audio[advance..].to_vec();
        }
    }

    if progressive {
        for token in &all_tokens[printed_tokens..] {
            print!("{}", token.text);
        }
        if !all_tokens.is_empty() {
            println!();
        }
    }

    let sentences = tokens_to_sentences(&all_tokens, &decoding_config.sentence);
    Ok(sentences_to_result(&sentences))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    let device = device(args.cpu)?;
    if args.debug {
        debug_log!(t0, "device={device:?}");
    }

    let api = HFClientSync::new()?;
    let (owner, name) = split_id(&args.model_id);
    let repo = api.model(owner, name);

    if args.debug {
        debug_log!(t0, "fetching config from {}", args.model_id);
    }
    let config_path = hub_file(&repo, "config.json").context("loading config.json")?;
    if args.debug {
        debug_log!(t0, "config loaded from {}", config_path.display());
    }
    let config: serde_json::Value = serde_json::from_reader(File::open(&config_path)?)?;

    if args.debug {
        debug_log!(t0, "fetching model weights from {}", args.model_id);
    }
    let safetensors_files = match hub_file(&repo, "model.safetensors.index.json") {
        Ok(index_path) => hub_load_safetensors(&repo, &index_path)?,
        Err(_) => vec![hub_file(&repo, "model.safetensors").context("loading model.safetensors")?],
    };
    if args.debug {
        let total_bytes: u64 = safetensors_files
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        debug_log!(
            t0,
            "{} weight file(s), {:.1} MB total",
            safetensors_files.len(),
            total_bytes as f64 / 1_048_576.0
        );
    }

    if args.debug {
        debug_log!(t0, "loading model into memory");
    }
    // SAFETY: The safetensors files are downloaded from HuggingFace Hub and are not
    // modified while memory-mapped. The mmap lifetime is tied to the VarBuilder.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, DType::F32, &device)? };
    let mut model = from_config_value(config, vb)?;
    if args.debug {
        debug_log!(t0, "model loaded successfully");
    }

    let decoding_config = DecodingConfig {
        debug_decode: args.debug,
        decoding: if let Some(beam_size) = args.beam_size {
            Decoding::Beam(Beam {
                beam_size,
                ..Beam::default()
            })
        } else {
            Decoding::Greedy(Greedy)
        },
        ..DecodingConfig::default()
    };

    if args.debug {
        print_debug(&mut model, &args.input)?;
    }

    if !args.stream
        && let Ok(meta) = std::fs::metadata(&args.input)
    {
        let size_mb = meta.len() as f64 / 1_048_576.0;
        if size_mb > 100.0 {
            eprintln!(
                "Warning: input file is {size_mb:.0} MB. \
                 Consider --stream for large files to reduce memory usage."
            );
        }
    }

    if args.debug {
        debug_log!(t0, "starting transcription of {}", args.input.display());
    }
    let result = if args.stream {
        if args.debug {
            debug_log!(t0, "streaming mode: chunk={:.1}s", args.stream_chunk_secs);
        }
        transcribe_streaming(&args, &mut model, &decoding_config, t0)?
    } else {
        model.transcribe(
            &args.input,
            &decoding_config,
            args.chunk_duration,
            args.overlap_duration,
            None,
        )?
    };

    if args.debug {
        debug_log!(t0, "transcription complete");
        let token_count = result.iter_tokens().count();
        debug_log!(
            t0,
            "text_len={} tokens={} sentences={}",
            result.text.len(),
            token_count,
            result.sentences.len()
        );
        for (i, tok) in result.iter_tokens().take(20).enumerate() {
            eprintln!(
                "         tok[{i}] id={} text={:?} start={:.3} dur={:.3} conf={:.3}",
                tok.id, tok.text, tok.start, tok.duration, tok.confidence
            );
        }
        if token_count > 20 {
            eprintln!("         ... ({} more tokens)", token_count - 20);
        }
    }

    let skip_print = args.stream && matches!(args.format, OutputFormat::Text) && !args.timestamps;
    if !skip_print {
        print_output(&args, &device, &result)?;
    }
    if args.debug {
        debug_log!(t0, "done");
    }
    Ok(())
}
