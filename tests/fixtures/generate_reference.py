#!/usr/bin/env python3
"""Generate golden reference outputs for the parakeet-candle parity tests.

The reference implementation is parakeet-mlx, which the Rust port follows
module by module and which reads the same `mlx-community` safetensors. MLX
has a CPU backend on Linux (`mlx[cpu]`), so this script runs on any x86-64
or arm64 machine.

One deliberate deviation from stock parakeet-mlx: its get_logmel takes the
STFT magnitude as |re| + |im| instead of sqrt(re^2 + im^2). NeMo, which the
model was trained with, uses the true magnitude, and so does the Rust port.
The script patches get_logmel to use the true magnitude everywhere (offline
and streaming) and records how much the stock behaviour differs in
reference.json under "parakeet_mlx_l1_magnitude".

Everything runs in float32. parakeet-mlx defaults to bfloat16 weights and
audio, which would hide real differences behind bf16 rounding noise.

Usage (from the repository root, in a throwaway venv outside the repo):

    uv venv -p 3.12 /tmp/parakeet-ref
    VIRTUAL_ENV=/tmp/parakeet-ref uv pip install -r tests/fixtures/requirements.txt
    /tmp/parakeet-ref/bin/python tests/fixtures/generate_reference.py

To rebuild the audio clip from the LibriVox source, add
`--source-mp3 art_of_war_01-02_sun_tzu_64kb.mp3` (needs ffmpeg). See
tests/fixtures/README.md for the source URL and checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten
from parakeet_mlx.utils import from_config
from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences
import parakeet_mlx.parakeet
from parakeet_mlx.audio import get_logmel as mlx_get_logmel
from parakeet_mlx.audio import hanning, stft
from parakeet_mlx.parakeet import Beam, DecodingConfig, Greedy

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
CLIP_SECONDS = 7.0
SOURCE_MP3_SHA256 = "6cb289fa9a13cfc94b18e8a89159da687c60db88c2b6377311f9bc5664d5492f"
# SHA-256 of the clip's samples as little-endian int16. FFmpeg n9.0.2 produced
# them; another FFmpeg build may decode or resample differently.
CLIP_PCM_SHA256 = "5f9c6ea6f9adf82a579b5a6cdcf08ffc68569280b490482905cee218f92abeab"

# Encoder frames whose full 1024-d vectors are stored verbatim.
ENCODER_ROW_FRAMES = [0, 1, 10, 25, 44, 60, 80, 87]

# Streaming configurations replayed by tests/reference_parity.rs:
# (left context, right context) in encoder frames, depth, chunk seconds.
# (256, 256) is the parakeet-mlx default; nothing is committed on this clip.
# (128, 16) commits frames but its left context covers all 88 encoder frames,
# so the rotating KV cache never wraps. (64, 16) wraps within the 7 s clip.
STREAM_CONFIGS = [((256, 256), 1, 0.5), ((128, 16), 1, 0.5), ((64, 16), 1, 0.5)]
BEAM_SIZE = 5

FIXTURE_DIR = Path(__file__).resolve().parent / "reference"
CLIP_PATH = FIXTURE_DIR / "librivox_7s.flac"


def get_logmel(x: mx.array, args) -> mx.array:
    """parakeet_mlx.audio.get_logmel with the true STFT magnitude.

    Only the magnitude line differs. The config in use (hann window,
    per_feature normalisation, no pad_to) is asserted in main().
    """
    if args.preemph is not None:
        x = mx.concat([x[:1], x[1:] - args.preemph * x[:-1]], axis=0)
    window = hanning(args.win_length).astype(x.dtype)
    x = stft(x, args.n_fft, args.hop_length, args.win_length, window)
    x = mx.abs(x)  # stock parakeet-mlx: |re| + |im|
    if args.mag_power != 1.0:
        x = mx.power(x, args.mag_power)
    x = mx.matmul(args._filterbanks.astype(x.dtype), x.T)
    x = mx.log(x + 1e-5)
    mean = mx.mean(x, axis=1, keepdims=True)
    std = mx.std(x, axis=1, keepdims=True)
    x = (x - mean) / (std + 1e-5)
    return mx.expand_dims(x.T, axis=0)


def make_clip(source_mp3: Path) -> None:
    """Decode, downmix, resample, and trim the LibriVox MP3 to 16 kHz s16."""
    digest = hashlib.sha256(source_mp3.read_bytes()).hexdigest()
    if digest != SOURCE_MP3_SHA256:
        raise SystemExit(f"unexpected source checksum {digest}")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "clip.wav"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-i", str(source_mp3),
                "-t", str(CLIP_SECONDS),
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                str(wav),
            ],
            check=True,
        )
        samples, sr = sf.read(wav, dtype="<i2")
    digest = hashlib.sha256(samples.tobytes()).hexdigest()
    if digest != CLIP_PCM_SHA256:
        raise SystemExit(
            f"decoded clip has PCM checksum {digest}, expected {CLIP_PCM_SHA256}; "
            "this FFmpeg build decodes or resamples differently from n9.0.2"
        )
    # libsndfile's FLAC encoder compresses this clip about 2x better than ffmpeg's.
    sf.write(CLIP_PATH, samples, sr, subtype="PCM_16")


def _local_window(s_q: int, s_k: int, w: int) -> tuple[mx.array, mx.array]:
    """Key index and validity for each (query, relative offset) pair.

    Matches the Metal kernels: queries stick to the right end of the keys, so
    query i sees key (s_k - s_q + i + r - w) for r in [0, 2w].
    """
    i = np.arange(s_q)[:, None]
    r = np.arange(2 * w + 1)[None, :]
    key = s_k - s_q + i + r - w
    valid = (key >= 0) & (key < s_k)
    return mx.array(np.clip(key, 0, s_k - 1)), mx.array(valid)


def _local_qk(self, q: mx.array, k: mx.array, w: int) -> mx.array:
    key, valid = _local_window(q.shape[2], k.shape[2], w)
    k_win = k[:, :, key, :]  # (B, H, S_q, 2w+1, D)
    scores = (mx.expand_dims(q, 3) * k_win).sum(axis=-1)
    return mx.where(valid, scores, -mx.inf).astype(q.dtype)


def _local_pv(self, prob: mx.array, v: mx.array, w: int) -> mx.array:
    key, valid = _local_window(prob.shape[2], v.shape[2], w)
    v_win = v[:, :, key, :]  # (B, H, S_p, 2w+1, D_v)
    weights = mx.where(valid, prob, 0.0)
    out = (mx.expand_dims(weights, -1) * v_win).sum(axis=3)  # (B, H, S_p, D_v)
    return out.transpose(0, 2, 1, 3).astype(prob.dtype)


def install_cpu_local_attention() -> None:
    """Replace parakeet-mlx's Metal-only local-attention kernels with MLX ops.

    Streaming switches the encoder to RelPositionMultiHeadLocalAttention, whose
    matmul_qk and matmul_pv call mx.fast.metal_kernel. These replacements
    compute the same banded products, including the out-of-range handling
    (-inf scores, zero contribution), with ordinary MLX ops that run on CPU.
    """
    from parakeet_mlx.attention import RelPositionMultiHeadLocalAttention

    RelPositionMultiHeadLocalAttention.matmul_qk = _local_qk
    RelPositionMultiHeadLocalAttention.matmul_pv = _local_pv


def load_model():
    """parakeet_mlx.from_pretrained, pinned to a revision and cast to float32."""
    config_path = hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
    weights_path = hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    with open(config_path) as f:
        config = json.load(f)
    model = from_config(config)
    model.load_weights(weights_path)
    params = [(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(params))
    return model, config


def token_dicts(tokens) -> list[dict]:
    return [
        {
            "id": int(t.id),
            "text": t.text,
            "start": float(t.start),
            "duration": float(t.duration),
            "confidence": float(t.confidence),
        }
        for t in tokens
    ]


def result_dict(result) -> dict:
    return {"text": result.text, "tokens": token_dicts(result.tokens)}


def decode(model, features, lengths, decoding) -> dict:
    config = DecodingConfig(decoding=decoding)
    hyps, _ = model.decode(features, lengths, config=config)
    return result_dict(sentences_to_result(tokens_to_sentences(hyps[0], config.sentence)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-mp3", type=Path, help="rebuild the audio clip first")
    args = parser.parse_args()

    if args.source_mp3 is not None:
        make_clip(args.source_mp3)

    mx.set_default_device(mx.cpu)
    install_cpu_local_attention()

    model, config = load_model()
    pre = model.preprocessor_config
    assert pre.window == "hann" and pre.normalize == "per_feature" and pre.pad_to == 0
    parakeet_mlx.parakeet.get_logmel = get_logmel

    audio_i16, sr = sf.read(CLIP_PATH, dtype="<i2")
    assert hashlib.sha256(audio_i16.tobytes()).hexdigest() == CLIP_PCM_SHA256
    assert sr == pre.sample_rate, (sr, pre.sample_rate)
    audio = (audio_i16.astype(np.float32) / 32768.0).astype(np.float32)
    audio_mx = mx.array(audio)

    # Log-mel features. The f16 copy is what the Rust tests read; the encoder
    # runs on that same rounded copy so its output is self-consistent.
    mel = get_logmel(audio_mx, pre)
    mx.eval(mel)
    mel_f16 = mel.astype(mx.float16)
    mel_in = mel_f16.astype(mx.float32)
    mel_np = np.array(mel)
    mel_rounding = float(np.abs(np.array(mel_in) - mel_np).max())

    features, lengths = model.encoder(mel_in)
    mx.eval(features, lengths)
    feats = np.array(features)[0]  # (T, D)
    n_frames = int(np.array(lengths)[0])
    assert feats.shape[0] == n_frames
    assert max(ENCODER_ROW_FRAMES) < n_frames

    greedy = decode(model, features, lengths, Greedy())
    beam = decode(model, features, lengths, Beam(beam_size=BEAM_SIZE))

    # End-to-end greedy from the unrounded mel, to show the f16 rounding of the
    # stored mel does not change the transcript.
    full_features, full_lengths = model.encoder(mel)
    greedy_full_mel = decode(model, full_features, full_lengths, Greedy())

    # Stock parakeet-mlx front end, for the record.
    l1_mel = mlx_get_logmel(audio_mx, pre)
    l1_features, l1_lengths = model.encoder(l1_mel)
    l1_greedy = decode(model, l1_features, l1_lengths, Greedy())
    l1_diff = np.abs(np.array(l1_mel) - mel_np)

    # Reference streaming: feed fixed-size chunks through StreamingParakeet.
    streaming = []
    for context, depth, chunk_seconds in STREAM_CONFIGS:
        chunk = int(chunk_seconds * pre.sample_rate)
        with model.transcribe_stream(
            context_size=context, depth=depth, decoding_config=DecodingConfig()
        ) as stream:
            for start in range(0, len(audio), chunk):
                stream.add_audio(audio_mx[start : start + chunk])
            streaming.append(
                {
                    "context_size": list(context),
                    "depth": depth,
                    "chunk_seconds": chunk_seconds,
                    **result_dict(stream.result),
                }
            )

    tensors = {
        "audio_head": mx.array(audio[:64]),
        "mel_f16": mel_f16,
        "encoder_rows": mx.array(feats[ENCODER_ROW_FRAMES]),
        "encoder_row_frames": mx.array(np.array(ENCODER_ROW_FRAMES, dtype=np.int32)),
        "encoder_frame_mean": mx.array(feats.mean(axis=1)),
        "encoder_frame_std": mx.array(feats.std(axis=1)),
        "encoder_channel_mean": mx.array(feats.mean(axis=0)),
    }
    mx.save_safetensors(str(FIXTURE_DIR / "tensors.safetensors"), tensors)

    summary = {
        "generator": "tests/fixtures/generate_reference.py",
        "versions": {
            pkg: version(pkg)
            for pkg in ["parakeet-mlx", "mlx", "mlx-cpu", "numpy", "librosa", "soundfile"]
        }
        | {"libsndfile": sf.__libsndfile_version__},
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "float32",
        "device": "cpu",
        # Lets the Rust mel test build PreprocessArgs without downloading the model.
        "preprocessor": config["preprocessor"],
        "audio": {
            "file": CLIP_PATH.name,
            "sample_rate": int(sr),
            "num_samples": int(len(audio)),
            "sum": float(audio.astype(np.float64).sum()),
            "sum_sq": float((audio.astype(np.float64) ** 2).sum()),
        },
        "mel": {
            "shape": list(mel.shape),
            "f16_rounding_max_abs": mel_rounding,
            "mean": float(mel_np.mean()),
            "std": float(mel_np.std()),
        },
        "encoder": {
            "shape": list(features.shape),
            "length": n_frames,
            "mean": float(feats.mean()),
            "std": float(feats.std()),
            "abs_max": float(np.abs(feats).max()),
        },
        "greedy": greedy,
        "greedy_full_precision_mel": greedy_full_mel,
        "beam": {"beam_size": BEAM_SIZE, **beam},
        "streaming": streaming,
        "parakeet_mlx_l1_magnitude": {
            "mel_max_abs_diff": float(l1_diff.max()),
            "mel_mean_abs_diff": float(l1_diff.mean()),
            "greedy": l1_greedy,
        },
    }
    with open(FIXTURE_DIR / "reference.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("greedy :", greedy["text"])
    print("beam   :", beam["text"])
    for s in streaming:
        print(f"stream {s['context_size']}:", s["text"])
    print("stock  :", l1_greedy["text"])
    print("full   :", greedy_full_mel["text"])


if __name__ == "__main__":
    main()
