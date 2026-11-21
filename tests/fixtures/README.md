# Reference fixtures

`tests/reference_parity.rs` compares parakeet-candle with
[parakeet-mlx](https://github.com/senstella/parakeet-mlx), the implementation
this crate follows module by module. The fixtures in `reference/` come from
`generate_reference.py`.

## Files

| File | Contents |
| --- | --- |
| `reference/librivox_7s.flac` | 7.0 s, 16 kHz mono s16 speech clip (117 KB). |
| `reference/tensors.safetensors` | Reference log-mel (f16), eight full encoder frames, and per-frame and per-channel encoder statistics (f32). |
| `reference/reference.json` | Versions, model revision, preprocessor config, audio checksums, and the greedy, beam, and streaming transcripts with token ids, timestamps, and confidences. |
| `generate_reference.py` | Generator. |
| `requirements.txt` | Exact Python package versions used for the committed fixtures. |

## Audio source

The clip is the first 7 seconds of chapters 1–2 of *The Art of War* by Sun
Tzu, read by Moira Fogarty for LibriVox. LibriVox recordings are in the
public domain. The clip says "This is a LibriVox recording. All LibriVox
recordings are in the public domain."

- Source: <https://archive.org/download/art_of_war_librivox/art_of_war_01-02_sun_tzu_64kb.mp3>
- SHA-256: `6cb289fa9a13cfc94b18e8a89159da687c60db88c2b6377311f9bc5664d5492f`

## Reference choices

- **Implementation:** parakeet-mlx 0.7.0 on MLX 0.32.4 with the CPU backend,
  so it runs on Linux. Model: `mlx-community/parakeet-tdt-0.6b-v3` at
  revision `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`.
- **Precision:** float32 weights and audio. parakeet-mlx defaults to bfloat16.
- **STFT magnitude:** stock parakeet-mlx computes `|re| + |im|`. NeMo, which
  trained the model, and parakeet-candle both use `sqrt(re² + im²)`. The
  generator patches parakeet-mlx to use the true magnitude. It still records
  the stock mel difference and transcript under `parakeet_mlx_l1_magnitude`.
- **Local attention on CPU:** streaming uses parakeet-mlx's local attention,
  whose kernels are Metal-only. The generator replaces them with equivalent
  MLX ops. With a context wider than the clip, the replacement matches full
  attention to 3e-7.
- **Encoder input:** the encoder runs on the f16-rounded mel that the Rust
  tests read. The reference encoder output is therefore exact for that input.

## Run the tests

The audio and log-mel tests run with `cargo test`. The model tests need the
weights (about 2.5 GB, downloaded once into the Hugging Face cache):

```sh
cargo test --release --test reference_parity -- --ignored --nocapture
```

## Regenerate the fixtures

Use a throwaway virtual environment outside the repository:

```sh
uv venv -p 3.12 /tmp/parakeet-ref
VIRTUAL_ENV=/tmp/parakeet-ref uv pip install -r tests/fixtures/requirements.txt
/tmp/parakeet-ref/bin/python -I tests/fixtures/generate_reference.py
```

To rebuild the clip from the source MP3 as well (needs ffmpeg), add
`--source-mp3 path/to/art_of_war_01-02_sun_tzu_64kb.mp3`. A run takes about
two minutes on a 16-core x86-64 CPU.
