# parakeet-candle

Transcribe audio locally with NVIDIA Parakeet models in Rust using [Candle](https://github.com/huggingface/candle). Supports TDT, RNNT, CTC, and hybrid TDT-CTC models and reads WAV, MP3, FLAC, Ogg Vorbis, and AAC (ADTS `.aac` or MP4/M4A) audio.

## Run

Install Rust with rustup to use the pinned toolchain. The first transcription downloads the model from Hugging Face Hub. Use `--model-id` to select a different repository with a compatible config and safetensors weights.

```sh
cargo run --locked --release -- --input audio.wav
cargo run --locked --release -- --help
```

For GPU inference, add `--features cuda` on NVIDIA or `--features metal,accelerate` on Apple Silicon before the first `--`. For Intel MKL on CPU, use `--features mkl`.

## Output

- Use `--format json --timestamps` for sentence and token timing and confidence scores. Times are in seconds.
- Use `--format srt` for subtitles.
- Use `--stream` to process a large file in chunks. Plain text prints progressively when timestamps are disabled.

The model implementation follows [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). Audio decoding uses [Symphonia](https://github.com/pdeljanov/Symphonia), and resampling uses [Rubato](https://github.com/HEnquist/rubato).
