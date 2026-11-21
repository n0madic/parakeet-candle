use audioadapter_buffers::direct::InterleavedSlice;
use candle::{DType, Device, Result, Tensor};
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use rubato::{
    Async, FixedAsync, FixedSync, Indexing, Resampler, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
use symphonia::core::errors::Error as DecodeError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PreprocessArgs {
    pub sample_rate: usize,
    pub normalize: String,
    pub window_size: f64,
    pub window_stride: f64,
    pub window: String,
    pub features: usize,
    pub n_fft: usize,
    pub dither: f64,
    #[serde(default)]
    pub pad_to: usize,
    #[serde(default)]
    pub pad_value: f64,
    #[serde(default = "default_preemph")]
    pub preemph: Option<f64>,
    #[serde(default = "default_mag_power")]
    pub mag_power: f64,
}

fn default_preemph() -> Option<f64> {
    Some(0.97)
}

fn default_mag_power() -> f64 {
    2.0
}

impl PreprocessArgs {
    pub fn win_length(&self) -> usize {
        (self.window_size * self.sample_rate as f64) as usize
    }

    pub fn hop_length(&self) -> usize {
        (self.window_stride * self.sample_rate as f64) as usize
    }

    /// Call after deserialization to catch invalid configs early.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(candle::Error::Msg("sample_rate must be > 0".to_string()));
        }
        if self.window_size <= 0.0 {
            return Err(candle::Error::Msg("window_size must be > 0".to_string()));
        }
        if self.window_stride <= 0.0 {
            return Err(candle::Error::Msg("window_stride must be > 0".to_string()));
        }
        if self.n_fft == 0 || !self.n_fft.is_power_of_two() {
            return Err(candle::Error::Msg(
                "n_fft must be a positive power of 2".to_string(),
            ));
        }
        if self.win_length() > self.n_fft {
            return Err(candle::Error::Msg(format!(
                "window length ({}) must not exceed n_fft ({})",
                self.win_length(),
                self.n_fft
            )));
        }
        if self.features == 0 {
            return Err(candle::Error::Msg("features must be > 0".to_string()));
        }
        Ok(())
    }
}

pub fn load_audio(path: &std::path::Path, sampling_rate: usize) -> Result<Vec<f32>> {
    let mut reader = AudioReader::open(path, sampling_rate)?;
    let mut audio = Vec::new();
    while !reader.is_finished() {
        audio.extend(reader.read_chunk(16_384)?);
    }

    if audio.is_empty() {
        return Err(candle::Error::Msg("no audio samples decoded".to_string()));
    }
    Ok(audio)
}

const RESAMPLE_CHUNK_SIZE: usize = 1024;

/// Minimum FFT output block, in frames. rubato's FFT resampler builds its
/// anti-aliasing filter on this block, so a short block rolls off well below
/// Nyquist. 512 frames keeps the response flat to 7.5 kHz at a 16 kHz target
/// for every input rate tested from 22.05 to 192 kHz, which covers the top of
/// Parakeet's mel filterbank.
const MIN_FFT_OUTPUT_BLOCK: usize = 512;

/// Largest FFT block allowed on either side. Rate pairs with no useful common
/// divisor (for example 44,101 Hz) would need huge blocks, so they use sinc
/// interpolation, which works for any ratio.
const MAX_FFT_BLOCK: usize = 16_384;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// FFT block sizes `(input, output)` for a rate pair, or `None` if the pair
/// needs blocks larger than [`MAX_FFT_BLOCK`]. Both sizes are even, since an
/// odd block makes rubato's reported delay off by a fraction of a sample.
fn fft_block_sizes(source_sr: usize, target_sr: usize) -> Option<(usize, usize)> {
    let g = gcd(source_sr, target_sr);
    let (min_in, min_out) = (source_sr / g, target_sr / g);
    let mut blocks = MIN_FFT_OUTPUT_BLOCK.div_ceil(min_out);
    if (blocks * min_in) % 2 == 1 || (blocks * min_out) % 2 == 1 {
        blocks *= 2;
    }
    let (fft_in, fft_out) = (blocks.checked_mul(min_in)?, blocks.checked_mul(min_out)?);
    (fft_in <= MAX_FFT_BLOCK && fft_out <= MAX_FFT_BLOCK).then_some((fft_in, fft_out))
}

fn make_resampler(
    source_sr: usize,
    target_sr: usize,
) -> std::result::Result<Box<dyn Resampler<f32>>, String> {
    let resampler: Box<dyn Resampler<f32>> = match fft_block_sizes(source_sr, target_sr) {
        // One sub-chunk of exactly `fft_in` frames makes rubato use these sizes.
        Some((fft_in, _)) => Box::new(
            rubato::Fft::new_custom(
                source_sr,
                target_sr,
                fft_in,
                1,
                1,
                WindowFunction::BlackmanHarris2,
                FixedSync::Input,
            )
            .map_err(|e| format!("resampler creation failed: {e}"))?,
        ),
        None => Box::new(sinc_resampler(source_sr, target_sr)?),
    };
    Ok(resampler)
}

fn sinc_resampler(source_sr: usize, target_sr: usize) -> std::result::Result<Async<f32>, String> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    Async::new_sinc(
        target_sr as f64 / source_sr as f64,
        2.0,
        &params,
        RESAMPLE_CHUNK_SIZE,
        1,
        FixedAsync::Input,
    )
    .map_err(|e| format!("resampler creation failed: {e}"))
}

/// Mono resampler fed incrementally. Output is aligned with the input: the
/// resampler's startup delay is dropped, and after [`finish`](Self::finish)
/// the total output length is `ceil(input_len * ratio)`.
struct StreamResampler {
    resampler: Box<dyn Resampler<f32>>,
    source_sr: usize,
    target_sr: usize,
    pending: Vec<f32>,
    scratch: Vec<f32>,
    delay_left: usize,
    input_len: usize,
    output_len: usize,
}

impl StreamResampler {
    fn new(source_sr: usize, target_sr: usize) -> Result<Self> {
        let resampler = make_resampler(source_sr, target_sr).map_err(candle::Error::Msg)?;
        Ok(Self {
            delay_left: resampler.output_delay(),
            scratch: vec![0.0; resampler.output_frames_max()],
            resampler,
            source_sr,
            target_sr,
            pending: Vec::new(),
            input_len: 0,
            output_len: 0,
        })
    }

    /// Resample every complete input chunk now available.
    fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        self.pending.extend_from_slice(input);
        self.input_len += input.len();
        let mut consumed = 0;
        while self.pending.len() - consumed >= self.resampler.input_frames_next() {
            let frames = self.resampler.input_frames_next();
            consumed += self.process(consumed, frames, None, out)?;
        }
        self.pending.drain(..consumed);
        Ok(())
    }

    /// Resample the remaining input, then feed silence until the delayed tail
    /// has been emitted.
    fn finish(&mut self, out: &mut Vec<f32>) -> Result<()> {
        let rest = self.pending.len();
        if rest > 0 {
            self.process(0, rest, Some(rest), out)?;
            self.pending.clear();
        }
        while self.output_len < self.expected_output_len() {
            self.process(0, 0, Some(0), out)?;
        }
        Ok(())
    }

    fn expected_output_len(&self) -> usize {
        let len =
            (self.input_len as u128 * self.target_sr as u128).div_ceil(self.source_sr as u128);
        len as usize
    }

    /// Run one resampler call on `pending[offset..offset + frames]`. Returns
    /// the number of input frames consumed.
    fn process(
        &mut self,
        offset: usize,
        frames: usize,
        partial_len: Option<usize>,
        out: &mut Vec<f32>,
    ) -> Result<usize> {
        let input = &self.pending[offset..offset + frames];
        let input = InterleavedSlice::new(input, 1, frames).map_err(candle::Error::wrap)?;
        let out_frames = self.scratch.len();
        let mut output = InterleavedSlice::new_mut(&mut self.scratch, 1, out_frames)
            .map_err(candle::Error::wrap)?;
        let indexing = Indexing {
            partial_len,
            ..Default::default()
        };
        let (consumed, produced) = self
            .resampler
            .process_into_buffer(&input, &mut output, Some(&indexing))
            .map_err(|e| candle::Error::Msg(format!("resampling failed: {e}")))?;

        let skip = self.delay_left.min(produced);
        self.delay_left -= skip;
        let keep = (produced - skip).min(self.expected_output_len() - self.output_len);
        out.extend_from_slice(&self.scratch[skip..skip + keep]);
        self.output_len += keep;
        Ok(consumed)
    }
}

/// Incremental audio file reader that decodes and resamples audio in chunks
/// without loading the entire file into memory.
pub struct AudioReader {
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    target_sr: usize,
    resampler: Option<StreamResampler>,
    mono_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    sample_buf: Vec<f32>,
    finished: bool,
    total_samples_read: usize,
}

impl AudioReader {
    pub fn open(path: &std::path::Path, target_sr: usize) -> Result<Self> {
        if !path.exists() {
            return Err(candle::Error::Msg(format!(
                "audio file not found: {}",
                path.display()
            )));
        }

        let file = std::fs::File::open(path)
            .map_err(|e| candle::Error::Msg(format!("failed to open audio file: {e}")))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let reader = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| candle::Error::Msg(format!("unsupported audio format: {e}")))?;

        let (track_id, codec_params) = reader
            .tracks()
            .iter()
            .filter_map(|track| Some((track.id, track.codec_params.as_ref()?.audio()?)))
            .find(|(_, params)| params.codec != CODEC_ID_NULL_AUDIO)
            .ok_or_else(|| candle::Error::Msg("no audio track found".to_string()))?;
        let source_sr = codec_params
            .sample_rate
            .ok_or_else(|| candle::Error::Msg("unknown sample rate".to_string()))?
            as usize;

        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
            .map_err(|e| candle::Error::Msg(format!("unsupported codec: {e}")))?;

        let resampler = if source_sr != target_sr {
            Some(StreamResampler::new(source_sr, target_sr)?)
        } else {
            None
        };

        Ok(Self {
            reader,
            decoder,
            track_id,
            target_sr,
            resampler,
            mono_buffer: Vec::new(),
            output_buffer: Vec::new(),
            sample_buf: Vec::new(),
            finished: false,
            total_samples_read: 0,
        })
    }

    /// Read the next chunk of resampled mono samples.
    /// Returns an empty `Vec` when all audio has been consumed.
    pub fn read_chunk(&mut self, requested_samples: usize) -> Result<Vec<f32>> {
        while self.output_buffer.len() < requested_samples && !self.finished {
            self.decode_next_packet()?;
            self.resample_buffered(false)?;
        }

        let n = requested_samples.min(self.output_buffer.len());
        let chunk: Vec<f32> = self.output_buffer.drain(..n).collect();
        self.total_samples_read += chunk.len();
        Ok(chunk)
    }

    pub fn is_finished(&self) -> bool {
        self.finished && self.output_buffer.is_empty()
    }

    /// Current read position in seconds (based on samples returned so far).
    pub fn position_secs(&self) -> f64 {
        self.total_samples_read as f64 / self.target_sr as f64
    }

    fn decode_next_packet(&mut self) -> Result<()> {
        let packet = match self.reader.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => {
                self.resample_buffered(true)?;
                self.finished = true;
                return Ok(());
            }
            Err(e) => return Err(candle::Error::Msg(format!("decode error: {e}"))),
        };
        if packet.track_id != self.track_id {
            return Ok(());
        }
        let decoded = match self.decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(DecodeError::IoError(_) | DecodeError::DecodeError(_)) => return Ok(()),
            Err(e) => return Err(candle::Error::Msg(format!("decode error: {e}"))),
        };
        decoded.copy_to_vec_interleaved(&mut self.sample_buf);
        let channels = decoded.spec().channels().count();

        if channels == 1 {
            self.mono_buffer.extend_from_slice(&self.sample_buf);
        } else {
            self.mono_buffer.extend(
                self.sample_buf
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32),
            );
        }

        Ok(())
    }

    fn resample_buffered(&mut self, flush: bool) -> Result<()> {
        let Some(resampler) = &mut self.resampler else {
            self.output_buffer.append(&mut self.mono_buffer);
            return Ok(());
        };
        resampler.push(&self.mono_buffer, &mut self.output_buffer)?;
        self.mono_buffer.clear();
        if flush {
            resampler.finish(&mut self.output_buffer)?;
        }
        Ok(())
    }
}

fn hz_to_mel(freq: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if freq < min_log_hz {
        freq / f_sp
    } else {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel < min_log_mel {
        f_sp * mel
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn mel_filterbank(sr: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let fmin = 0.0;
    let fmax = sr as f64 / 2.0;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let t = i as f64 / (n_mels + 1) as f64;
        mel_points.push(mel_min + (mel_max - mel_min) * t);
    }

    let hz_points: Vec<f64> = mel_points.into_iter().map(mel_to_hz).collect();
    let bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((n_fft + 1) as f64 * hz / sr as f64).floor() as usize)
        .collect();

    let n_fft_bins = n_fft / 2 + 1;
    let mut filters = vec![0f32; n_mels * n_fft_bins];
    for m in 0..n_mels {
        let f_m_minus = bins[m];
        let f_m = bins[m + 1];
        let f_m_plus = bins[m + 2];

        if f_m_minus == f_m || f_m == f_m_plus {
            continue;
        }

        for k in f_m_minus..f_m {
            if k < n_fft_bins {
                filters[m * n_fft_bins + k] =
                    (k as f64 - f_m_minus as f64) as f32 / (f_m as f64 - f_m_minus as f64) as f32;
            }
        }
        for k in f_m..f_m_plus {
            if k < n_fft_bins {
                filters[m * n_fft_bins + k] =
                    (f_m_plus as f64 - k as f64) as f32 / (f_m_plus as f64 - f_m as f64) as f32;
            }
        }
    }

    // Slaney-style normalization
    for m in 0..n_mels {
        let f_m_minus = hz_points[m];
        let f_m_plus = hz_points[m + 2];
        let enorm = 2.0 / (f_m_plus - f_m_minus).max(1e-6);
        for k in 0..n_fft_bins {
            filters[m * n_fft_bins + k] *= enorm as f32;
        }
    }

    filters
}

fn window_values(kind: &str, len: usize) -> Vec<f32> {
    match kind {
        // Periodic form: 2π·i/N (matches torch.hann_window(periodic=True) / NeMo)
        "hann" | "hanning" => (0..len)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / len as f32).cos())
            .collect(),
        "hamming" => (0..len)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / len as f32).cos())
            .collect(),
        "blackman" => (0..len)
            .map(|i| {
                0.42 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / len as f32).cos()
                    + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / len as f32).cos()
            })
            .collect(),
        "bartlett" => (0..len)
            .map(|i| {
                let v = (i as f32 - (len as f32 - 1.0) / 2.0).abs();
                1.0 - v / ((len as f32 - 1.0) / 2.0)
            })
            .collect(),
        _ => vec![1.0; len],
    }
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if samples.is_empty() || pad == 0 {
        return samples.to_vec();
    }
    if samples.len() < 2 {
        let mut out = Vec::with_capacity(samples.len() + 2 * pad);
        out.extend(std::iter::repeat_n(samples[0], pad));
        out.extend_from_slice(samples);
        out.extend(std::iter::repeat_n(samples[0], pad));
        return out;
    }
    let mut out = Vec::with_capacity(samples.len() + 2 * pad);
    let prefix = samples[1..=pad.min(samples.len() - 1)]
        .iter()
        .rev()
        .cloned();
    let suffix = samples[samples.len().saturating_sub(pad + 1)..samples.len() - 1]
        .iter()
        .rev()
        .cloned();
    out.extend(prefix);
    out.extend_from_slice(samples);
    out.extend(suffix);
    out
}

/// Magnitude of one FFT bin raised to `power`. The `power == 2` case skips the
/// square root.
fn spectral_magnitude(bin: Complex<f32>, power: f64) -> f32 {
    if (power - 2.0).abs() <= f64::EPSILON {
        bin.norm_sqr()
    } else if (power - 1.0).abs() <= f64::EPSILON {
        bin.norm()
    } else {
        bin.norm().powf(power as f32)
    }
}

pub fn get_logmel(samples: &[f32], args: &PreprocessArgs, device: &Device) -> Result<Tensor> {
    let mut audio = samples.to_vec();
    if args.pad_to > 0 && audio.len() < args.pad_to {
        audio.resize(args.pad_to, args.pad_value as f32);
    }

    if let Some(preemph) = args.preemph
        && audio.len() > 1
    {
        let mut emphasized = Vec::with_capacity(audio.len());
        emphasized.push(audio[0]);
        for i in 1..audio.len() {
            emphasized.push(audio[i] - preemph as f32 * audio[i - 1]);
        }
        audio = emphasized;
    }

    let win_length = args.win_length();
    let hop_length = args.hop_length();
    let n_fft = args.n_fft;

    let window = window_values(&args.window, win_length);
    let pad = n_fft / 2;
    let padded = reflect_pad(&audio, pad);

    let frame_count = if padded.len() < win_length {
        0
    } else {
        (padded.len() - win_length + hop_length) / hop_length
    };

    let n_fft_bins = n_fft / 2 + 1;
    let filters = mel_filterbank(args.sample_rate, n_fft, args.features);

    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n_fft);
    let mut frame_buf = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    let mut mags = vec![0f32; n_fft_bins];

    let mut mel = vec![0f32; args.features * frame_count];
    for frame in 0..frame_count {
        let start = frame * hop_length;
        let slice = &padded[start..start + win_length];
        for ((dst, &v), &w) in frame_buf.iter_mut().zip(slice).zip(&window) {
            *dst = v * w;
        }
        frame_buf[win_length..].fill(0.0);

        fft.process_with_scratch(&mut frame_buf, &mut spectrum, &mut scratch)
            .map_err(candle::Error::wrap)?;
        for (mag, bin) in mags.iter_mut().zip(&spectrum) {
            *mag = spectral_magnitude(*bin, args.mag_power);
        }

        for mel_idx in 0..args.features {
            let filter = &filters[mel_idx * n_fft_bins..(mel_idx + 1) * n_fft_bins];
            let sum: f32 = filter.iter().zip(&mags).map(|(f, m)| f * m).sum();
            mel[mel_idx * frame_count + frame] = (sum + 1e-5).ln();
        }
    }

    if args.normalize == "per_feature" {
        for mel_idx in 0..args.features {
            let offset = mel_idx * frame_count;
            let slice = &mel[offset..offset + frame_count];
            let mean = slice.iter().sum::<f32>() / frame_count.max(1) as f32;
            let var =
                slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / frame_count.max(1) as f32;
            let std = var.sqrt();
            for v in &mut mel[offset..offset + frame_count] {
                *v = (*v - mean) / (std + 1e-5);
            }
        }
    } else {
        let mean = mel.iter().sum::<f32>() / mel.len().max(1) as f32;
        let var = mel.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / mel.len().max(1) as f32;
        let std = var.sqrt();
        for v in &mut mel {
            *v = (*v - mean) / (std + 1e-5);
        }
    }

    // shape: (features, frames) -> (frames, features)
    let mut mel_t = vec![0f32; mel.len()];
    for mel_idx in 0..args.features {
        for frame in 0..frame_count {
            mel_t[frame * args.features + mel_idx] = mel[mel_idx * frame_count + frame];
        }
    }

    let mel_tensor = Tensor::from_vec(mel_t, (frame_count, args.features), device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?;
    Ok(mel_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Power spectrum by direct DFT, used as an independent reference.
    fn naive_power_spectrum(signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        (0..=n / 2)
            .map(|k| {
                let (mut re, mut im) = (0f64, 0f64);
                for (i, &x) in signal.iter().enumerate() {
                    let theta = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                    re += x as f64 * theta.cos();
                    im += x as f64 * theta.sin();
                }
                (re * re + im * im) as f32
            })
            .collect()
    }

    #[test]
    fn realfft_power_matches_naive_dft() {
        let n = 512;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 3100.0 * t).cos()
                    + 0.01 * ((i * 7919) % 101) as f32
            })
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
        let mut input = signal.clone();
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut input, &mut spectrum).unwrap();
        let got: Vec<f32> = spectrum
            .iter()
            .map(|&bin| spectral_magnitude(bin, 2.0))
            .collect();
        let want = naive_power_spectrum(&signal);
        assert_eq!(got.len(), want.len());
        let peak = want.iter().cloned().fold(0f32, f32::max);
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-4 * peak, "bin {k}: got {g}, want {w}");
        }
    }

    #[test]
    fn spectral_magnitude_powers() {
        let bin = Complex::new(3.0f32, 4.0);
        assert_eq!(spectral_magnitude(bin, 1.0), 5.0);
        assert_eq!(spectral_magnitude(bin, 2.0), 25.0);
        assert!((spectral_magnitude(bin, 3.0) - 125.0).abs() < 1e-3);
    }

    #[test]
    fn mel_filterbank_shape() {
        let sr = 16000;
        let n_fft = 512;
        let n_mels = 80;
        let filters = mel_filterbank(sr, n_fft, n_mels);
        assert_eq!(filters.len(), n_mels * (n_fft / 2 + 1));
    }

    #[test]
    fn mel_filterbank_non_negative() {
        let filters = mel_filterbank(16000, 512, 80);
        for &v in &filters {
            assert!(v >= 0.0, "filter value = {v}");
        }
    }

    #[test]
    fn hz_mel_roundtrip() {
        for &freq in &[0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(freq);
            let hz_back = mel_to_hz(mel);
            assert!(
                (hz_back - freq).abs() < 1e-6,
                "roundtrip failed for {freq}: got {hz_back}"
            );
        }
    }

    #[test]
    fn reflect_pad_basic() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&samples, 2);
        assert_eq!(padded[0], 3.0);
        assert_eq!(padded[1], 2.0);
        assert_eq!(&padded[2..7], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(padded[7], 4.0);
        assert_eq!(padded[8], 3.0);
    }

    #[test]
    fn reflect_pad_empty() {
        let padded = reflect_pad(&[], 5);
        assert!(padded.is_empty());
    }

    #[test]
    fn reflect_pad_single_element() {
        let padded = reflect_pad(&[42.0], 3);
        assert_eq!(padded, vec![42.0; 7]);
    }

    #[test]
    fn window_hann_endpoints() {
        let w = window_values("hann", 8);
        assert_eq!(w.len(), 8);
        assert!(w[0].abs() < 1e-6, "hann[0] should be ~0");
    }

    #[test]
    fn window_unknown_returns_ones() {
        let w = window_values("unknown_window", 4);
        assert_eq!(w, vec![1.0; 4]);
    }

    fn valid_preprocess_args() -> PreprocessArgs {
        PreprocessArgs {
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
        }
    }

    #[test]
    fn preprocess_args_valid() {
        valid_preprocess_args().validate().unwrap();
    }

    #[test]
    fn preprocess_args_zero_sample_rate() {
        let mut args = valid_preprocess_args();
        args.sample_rate = 0;
        assert!(args.validate().is_err());
    }

    #[test]
    fn preprocess_args_non_power_of_two_nfft() {
        let mut args = valid_preprocess_args();
        args.n_fft = 400;
        assert!(args.validate().is_err());
    }

    #[test]
    fn preprocess_args_window_longer_than_nfft() {
        let mut args = valid_preprocess_args();
        args.window_size = 0.064; // 1024 samples at 16 kHz
        assert!(args.validate().is_err());
    }

    #[test]
    fn preprocess_args_zero_features() {
        let mut args = valid_preprocess_args();
        args.features = 0;
        assert!(args.validate().is_err());
    }

    #[test]
    fn get_logmel_produces_correct_shape() {
        let sr = 16000;
        let args = PreprocessArgs {
            sample_rate: sr,
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
        let audio = vec![0.0f32; sr];
        let device = Device::Cpu;
        let mel = get_logmel(&audio, &args, &device).unwrap();
        let (b, t, f) = mel.dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(f, 80);
        assert!(t > 0, "expected positive time frames, got {t}");
    }

    fn sine(freq: f64, sr: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect()
    }

    fn resample_in_pieces(input: &[f32], src: usize, dst: usize, piece: usize) -> Vec<f32> {
        let mut resampler = StreamResampler::new(src, dst).unwrap();
        let mut out = Vec::new();
        for chunk in input.chunks(piece.max(1)) {
            resampler.push(chunk, &mut out).unwrap();
        }
        resampler.finish(&mut out).unwrap();
        out
    }

    /// Peak amplitude of a unit sine at `freq` after resampling, measured away
    /// from the clip edges.
    fn tone_gain(src: usize, dst: usize, freq: f64) -> f64 {
        let out = resample_in_pieces(&sine(freq, src, src), src, dst, 4096);
        let mid = &out[dst / 4..out.len() - dst / 4];
        let mean_sq = mid.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / mid.len() as f64;
        (2.0 * mean_sq).sqrt()
    }

    /// Resampling a 1 kHz sine must give the same sine at the new rate: exact
    /// length, no startup delay, no drift. `max_lag` is the allowed phase
    /// offset in output samples. Edges are skipped because the filter sees
    /// silence beyond the clip.
    fn assert_resampled_sine(src: usize, dst: usize, max_lag: f32) {
        let input = sine(1000.0, src, 2 * src);
        let out = resample_in_pieces(&input, src, dst, 1000);
        assert_eq!(out.len(), 2 * dst, "{src} -> {dst}: output length");
        let want = sine(1000.0, dst, out.len());
        let edge = dst / 4;
        let max_err = out[edge..out.len() - edge]
            .iter()
            .zip(&want[edge..])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // A phase offset of `lag` samples gives a peak error of about
        // 2*pi*f*lag/dst for a unit sine.
        let limit = 2.0 * std::f32::consts::PI * 1000.0 * max_lag / dst as f32;
        assert!(
            max_err < limit,
            "{src} -> {dst}: max error {max_err}, limit {limit}"
        );
    }

    #[test]
    fn fft_resampler_is_aligned() {
        // 7 and 8 kHz upsample. 11,025 and 22,050 Hz have odd minimum blocks.
        for src in [7000, 8000, 11025, 22050, 32000, 44100, 48000, 96000, 192000] {
            assert!(
                fft_block_sizes(src, 16000).is_some(),
                "{src} should use FFT"
            );
            assert_resampled_sine(src, 16000, 0.05);
        }
    }

    #[test]
    fn fft_resampler_passband_and_stopband() {
        for src in [22050, 44100, 48000, 96000, 192000] {
            for freq in [1000.0, 4000.0, 7000.0, 7500.0] {
                let gain = tone_gain(src, 16000, freq);
                assert!(gain > 0.98, "{src} Hz: {freq} Hz tone kept {gain:.3}");
            }
            // 9 kHz is above the 8 kHz output Nyquist and would alias to 7 kHz.
            let alias = tone_gain(src, 16000, 9000.0);
            assert!(alias < 0.01, "{src} Hz: 9 kHz alias {alias:.4}");
        }
    }

    #[test]
    fn fft_block_sizes_are_even_and_bounded() {
        for src in [7000, 8000, 11025, 22050, 44100, 48000, 192000] {
            let (fft_in, fft_out) = fft_block_sizes(src, 16000).unwrap();
            assert_eq!(fft_in % 2, 0, "{src}: input block {fft_in}");
            assert_eq!(fft_out % 2, 0, "{src}: output block {fft_out}");
            assert!(fft_out >= MIN_FFT_OUTPUT_BLOCK);
        }
        // No useful common divisor, or an extreme ratio: use sinc instead.
        assert_eq!(fft_block_sizes(44101, 16000), None);
        assert_eq!(fft_block_sizes(1, 16000), None);
    }

    #[test]
    fn sinc_fallback_is_aligned() {
        assert_resampled_sine(44101, 16000, 1.0);
        assert_resampled_sine(8463, 16000, 1.0);
    }

    #[test]
    fn output_length_is_exact() {
        // 8463 * 16000 / 8463 is exactly 16000; float rounding gave 16001.
        assert_eq!(
            resample_in_pieces(&vec![0.0; 8463], 8463, 16000, 1000).len(),
            16000
        );
        assert_eq!(resample_in_pieces(&[0.5; 10], 44100, 16000, 10).len(), 4);
        assert!(resample_in_pieces(&[], 44100, 16000, 1).is_empty());
    }

    #[test]
    fn output_independent_of_push_sizes() {
        let input = sine(300.0, 44100, 30_000);
        let a = resample_in_pieces(&input, 44100, 16000, 1);
        let b = resample_in_pieces(&input, 44100, 16000, 4093);
        let c = resample_in_pieces(&input, 44100, 16000, input.len());
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        for (x, (y, z)) in a.iter().zip(b.iter().zip(&c)) {
            assert!((x - y).abs() < 1e-5 && (x - z).abs() < 1e-5);
        }
    }
}
