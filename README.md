# parakeet-candle

Offline automatic speech recognition (ASR) in Rust, built on the [Candle](https://github.com/huggingface/candle) ML framework. A standalone port of NVIDIA's [Parakeet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html) ASR models that runs entirely locally with no external service dependencies.

## Features

- **Multiple model architectures** -- TDT (Token-and-Duration Transducer), RNNT, CTC, and hybrid TDT-CTC
- **Audio format support** -- WAV, MP3, FLAC, OGG/Vorbis, AAC
- **Output formats** -- plain text, JSON with token-level details, SRT subtitles
- **Timestamps and confidence scores** -- sentence and token-level alignment with entropy-based confidence
- **Streaming mode** -- progressive transcription for large files with lower memory usage
- **Beam search decoding** -- configurable beam size with length normalization
- **GPU acceleration** -- CUDA, Apple Metal, Intel MKL; automatic fallback to CPU

## Installation

### Prerequisites

- Rust (edition 2024 or later)

### Build

```bash
# CPU-only
cargo build --release

# Apple Silicon (recommended on macOS)
cargo build --release --features metal,accelerate

# NVIDIA GPU
cargo build --release --features cuda

# Intel MKL
cargo build --release --features mkl
```

## Usage

Models are downloaded automatically from Hugging Face Hub on first run.

```bash
# Basic transcription
parakeet-candle --input audio.wav

# Force CPU
parakeet-candle --input audio.wav --cpu

# With sentence timestamps
parakeet-candle --input audio.wav --timestamps

# JSON output
parakeet-candle --input audio.wav --format json --timestamps

# SRT subtitles
parakeet-candle --input audio.wav --format srt

# Beam search decoding
parakeet-candle --input audio.wav --beam-size 5

# Streaming mode for large files
parakeet-candle --input long-recording.wav --stream --stream-chunk-secs 30

# Custom model
parakeet-candle --input audio.wav --model-id nvidia/parakeet-tdt-0.6b-v2
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *required* | Path to audio file |
| `--model-id` | `mlx-community/parakeet-tdt-0.6b-v3` | Hugging Face model ID |
| `--cpu` | `false` | Force CPU inference |
| `--beam-size` | greedy | Enable beam search with given width |
| `--timestamps` | `false` | Print sentence-level timestamps |
| `--format` | `text` | Output format: `text`, `json`, `srt` |
| `--stream` | `false` | Streaming mode for large files |
| `--stream-chunk-secs` | `30.0` | Chunk size per streaming iteration (seconds) |
| `--chunk-duration` | disabled | Chunk duration for batch processing (seconds) |
| `--overlap-duration` | `15.0` | Overlap between chunks (seconds) |
| `--debug` | `false` | Print decoder internals and token alignment data |

### JSON output

When using `--format json --timestamps`, the output includes token-level detail:

```json
{
  "text": "the full transcript",
  "model_id": "mlx-community/parakeet-tdt-0.6b-v3",
  "input": "audio.wav",
  "device": "metal",
  "sentences": [
    {
      "text": "sentence text",
      "start": 0.0,
      "end": 2.5,
      "duration": 2.5,
      "confidence": 0.95,
      "tokens": [
        {
          "text": "sentence",
          "start": 0.0,
          "end": 0.8,
          "duration": 0.8,
          "confidence": 0.97
        }
      ]
    }
  ]
}
```

## Architecture

```
src/
  main.rs              # CLI entry point and output formatting
  lib.rs               # Library root
  parakeet/
    model.rs           # Model loading and config deserialization
    conformer.rs       # Conformer encoder (self-attention + convolution)
    attention.rs       # Multi-head relative-position attention
    rnnt.rs            # RNNT/TDT decoder and joint network
    ctc.rs             # CTC decoder
    alignment.rs       # Token and sentence alignment from frame-level output
    audio.rs           # Audio decoding, resampling, and log-mel spectrogram
    cache.rs           # KV cache for streaming inference
    tokenizer.rs       # BPE token decoding
```

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) -- Hugging Face's minimalist ML framework for Rust
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) -- original Parakeet model implementation
- [Symphonia](https://github.com/pdeljanov/Symphonia) -- pure Rust audio decoding
- [Rubato](https://github.com/HEnquist/rubato) -- sample rate conversion
