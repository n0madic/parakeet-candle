use candle::{DType, Device, Result, Tensor};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

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

    /// Validate that all fields have sensible values.
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
        if self.features == 0 {
            return Err(candle::Error::Msg("features must be > 0".to_string()));
        }
        Ok(())
    }
}

pub fn load_audio(path: &std::path::Path, sampling_rate: usize) -> Result<Vec<f32>> {
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

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| candle::Error::Msg(format!("unsupported audio format: {e}")))?;
    let mut reader = probed.format;

    let track = reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| candle::Error::Msg("no audio track found".to_string()))?;
    let track_id = track.id;
    let source_sr = track
        .codec_params
        .sample_rate
        .ok_or_else(|| candle::Error::Msg("unknown sample rate".to_string()))?
        as usize;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let codec_params = track.codec_params.clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| candle::Error::Msg(format!("unsupported codec: {e}")))?;

    let mut raw_samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match reader.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(candle::Error::Msg(format!("decode error: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(_))
            | Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(candle::Error::Msg(format!("decode error: {e}"))),
        };
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let buf = sample_buf.get_or_insert_with(|| SampleBuffer::<f32>::new(duration, spec));
        buf.copy_interleaved_ref(decoded);
        raw_samples.extend_from_slice(buf.samples());
    }

    if raw_samples.is_empty() {
        return Err(candle::Error::Msg("no audio samples decoded".to_string()));
    }

    // Convert to mono by averaging channels
    let mono = if channels <= 1 {
        raw_samples
    } else {
        raw_samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Skip resampling if already at target rate
    if source_sr == sampling_rate {
        return Ok(mono);
    }

    // Sinc resampling via rubato
    let mut resampler = sinc_resampler(source_sr, sampling_rate).map_err(candle::Error::Msg)?;

    let mut output =
        Vec::with_capacity(mono.len() * sampling_rate / source_sr + RESAMPLE_CHUNK_SIZE);
    for chunk in mono.chunks(RESAMPLE_CHUNK_SIZE) {
        let result = if chunk.len() == RESAMPLE_CHUNK_SIZE {
            resampler.process(&[chunk], None)
        } else {
            resampler.process_partial(Some(&[chunk]), None)
        };
        let waves = result.map_err(|e| candle::Error::Msg(format!("resampling failed: {e}")))?;
        output.extend_from_slice(&waves[0]);
    }

    Ok(output)
}

const RESAMPLE_CHUNK_SIZE: usize = 1024;

fn sinc_resampler(
    source_sr: usize,
    target_sr: usize,
) -> std::result::Result<SincFixedIn<f32>, String> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    SincFixedIn::<f32>::new(
        target_sr as f64 / source_sr as f64,
        2.0,
        params,
        RESAMPLE_CHUNK_SIZE,
        1,
    )
    .map_err(|e| format!("resampler creation failed: {e}"))
}

/// Incremental audio file reader that decodes and resamples audio in chunks
/// without loading the entire file into memory.
pub struct AudioReader {
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    channels: usize,
    target_sr: usize,
    resampler: Option<SincFixedIn<f32>>,
    mono_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    sample_buf: Option<SampleBuffer<f32>>,
    finished: bool,
    total_samples_read: usize,
}

impl AudioReader {
    /// Open an audio file for incremental reading at the given target sample rate.
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

        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| candle::Error::Msg(format!("unsupported audio format: {e}")))?;
        let reader = probed.format;

        let track = reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| candle::Error::Msg("no audio track found".to_string()))?;
        let track_id = track.id;
        let source_sr = track
            .codec_params
            .sample_rate
            .ok_or_else(|| candle::Error::Msg("unknown sample rate".to_string()))?
            as usize;
        let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
        let codec_params = track.codec_params.clone();

        let decoder = symphonia::default::get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| candle::Error::Msg(format!("unsupported codec: {e}")))?;

        let resampler = if source_sr != target_sr {
            Some(sinc_resampler(source_sr, target_sr).map_err(candle::Error::Msg)?)
        } else {
            None
        };

        Ok(Self {
            reader,
            decoder,
            track_id,
            channels,
            target_sr,
            resampler,
            mono_buffer: Vec::new(),
            output_buffer: Vec::new(),
            sample_buf: None,
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

    /// Returns `true` when all audio has been consumed and output buffer is empty.
    pub fn is_finished(&self) -> bool {
        self.finished && self.output_buffer.is_empty()
    }

    /// Current read position in seconds (based on samples returned so far).
    pub fn position_secs(&self) -> f64 {
        self.total_samples_read as f64 / self.target_sr as f64
    }

    fn decode_next_packet(&mut self) -> Result<()> {
        let packet = match self.reader.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                // End of file: flush resampler
                self.resample_buffered(true)?;
                self.finished = true;
                return Ok(());
            }
            Err(e) => return Err(candle::Error::Msg(format!("decode error: {e}"))),
        };
        if packet.track_id() != self.track_id {
            return Ok(());
        }
        let decoded = match self.decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(_))
            | Err(symphonia::core::errors::Error::DecodeError(_)) => return Ok(()),
            Err(e) => return Err(candle::Error::Msg(format!("decode error: {e}"))),
        };
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let buf = self
            .sample_buf
            .get_or_insert_with(|| SampleBuffer::<f32>::new(duration, spec));
        buf.copy_interleaved_ref(decoded);
        let raw = buf.samples();

        // Convert to mono
        if self.channels <= 1 {
            self.mono_buffer.extend_from_slice(raw);
        } else {
            self.mono_buffer.extend(
                raw.chunks_exact(self.channels)
                    .map(|frame| frame.iter().sum::<f32>() / self.channels as f32),
            );
        }

        Ok(())
    }

    fn resample_buffered(&mut self, flush: bool) -> Result<()> {
        let Some(resampler) = &mut self.resampler else {
            // No resampling needed: move mono directly to output
            self.output_buffer.append(&mut self.mono_buffer);
            return Ok(());
        };

        // Process complete chunks
        while self.mono_buffer.len() >= RESAMPLE_CHUNK_SIZE {
            let chunk: Vec<f32> = self.mono_buffer.drain(..RESAMPLE_CHUNK_SIZE).collect();
            let waves = resampler
                .process(&[&chunk], None)
                .map_err(|e| candle::Error::Msg(format!("resampling failed: {e}")))?;
            self.output_buffer.extend_from_slice(&waves[0]);
        }

        // Flush remaining partial chunk at end of file
        if flush && !self.mono_buffer.is_empty() {
            let remaining: Vec<f32> = self.mono_buffer.drain(..).collect();
            let waves = resampler
                .process_partial(Some(&[&remaining]), None)
                .map_err(|e| candle::Error::Msg(format!("resampling failed: {e}")))?;
            self.output_buffer.extend_from_slice(&waves[0]);
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

/// Compute the FFT of `inp`. Non-power-of-2 lengths are zero-padded to the
/// next power of 2 so the recursive radix-2 path is always used (O(n log n)).
fn fft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![inp[0], 0.0];
    }
    let n_padded = n.next_power_of_two();
    if n_padded != n {
        let mut padded = vec![0f32; n_padded];
        padded[..n].copy_from_slice(inp);
        return fft_radix2(&padded);
    }
    fft_radix2(inp)
}

/// Radix-2 Cooley-Tukey FFT. Input length MUST be a power of 2.
fn fft_radix2(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    debug_assert!(n.is_power_of_two(), "fft_radix2 requires power-of-2 length");
    if n == 1 {
        return vec![inp[0], 0.0];
    }
    let mut out = vec![0f32; n * 2];

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for (i, &value) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(value);
        } else {
            odd.push(value);
        }
    }

    let even_fft = fft_radix2(&even);
    let odd_fft = fft_radix2(&odd);

    let two_pi = std::f32::consts::PI * 2.0;
    let n_t = n as f32;
    for k in 0..n / 2 {
        let k_t = k as f32;
        let theta = two_pi * k_t / n_t;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
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

    let mut mel = vec![0f32; args.features * frame_count];
    for frame in 0..frame_count {
        let start = frame * hop_length;
        let mut frame_buf = vec![0f32; n_fft];
        let slice = &padded[start..start + win_length];
        for (i, &v) in slice.iter().enumerate() {
            frame_buf[i] = v * window[i];
        }

        let fft_out = fft(&frame_buf);
        let mut mags = vec![0f32; n_fft_bins];
        for k in 0..n_fft_bins {
            let re = fft_out[2 * k];
            let im = fft_out[2 * k + 1];
            let mut mag = re.hypot(im);
            if (args.mag_power - 1.0).abs() > f64::EPSILON {
                mag = mag.powf(args.mag_power as f32);
            }
            mags[k] = mag;
        }

        for mel_idx in 0..args.features {
            let mut sum = 0.0f32;
            let filter_offset = mel_idx * n_fft_bins;
            for k in 0..n_fft_bins {
                sum += filters[filter_offset + k] * mags[k];
            }
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

    #[test]
    fn fft_dc_signal() {
        // FFT of a constant signal should have all energy in bin 0.
        let signal = vec![1.0f32; 8];
        let result = fft(&signal);
        // Bin 0: real = 8.0, imag = 0.0
        assert!((result[0] - 8.0).abs() < 1e-5);
        assert!(result[1].abs() < 1e-5);
        // All other bins should be zero
        for i in 1..8 {
            assert!(result[2 * i].abs() < 1e-4, "bin {i} re = {}", result[2 * i]);
            assert!(
                result[2 * i + 1].abs() < 1e-4,
                "bin {i} im = {}",
                result[2 * i + 1]
            );
        }
    }

    #[test]
    fn fft_single_cosine() {
        // cos(2*pi*k*n/N) for k=1, N=8: energy in bins 1 and 7
        let n = 8;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect();
        let result = fft(&signal);
        // Bin 1 should have magnitude N/2 = 4
        let mag_bin1 = (result[2] * result[2] + result[3] * result[3]).sqrt();
        assert!(
            (mag_bin1 - 4.0).abs() < 1e-3,
            "bin 1 magnitude = {mag_bin1}"
        );
        // Bin 0 should be ~0
        assert!(result[0].abs() < 1e-3, "bin 0 = {}", result[0]);
    }

    #[test]
    fn fft_odd_length_zero_padded() {
        // Non-power-of-2 is zero-padded to next power of 2
        let signal = vec![1.0, 2.0, 3.0];
        let result = fft(&signal);
        // Padded to length 4: [1, 2, 3, 0]
        // X[0] = 1+2+3+0 = 6
        assert!((result[0] - 6.0).abs() < 1e-4);
        // Output has 4 complex values (8 floats)
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn fft_empty_input() {
        let result = fft(&[]);
        assert!(result.is_empty());
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
        // Left reflection of [1..=2] reversed = [3, 2]
        assert_eq!(padded[0], 3.0);
        assert_eq!(padded[1], 2.0);
        // Original
        assert_eq!(&padded[2..7], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Right reflection of [3..4] reversed = [4, 3]
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
        // 1 second of silence
        let audio = vec![0.0f32; sr];
        let device = Device::Cpu;
        let mel = get_logmel(&audio, &args, &device).unwrap();
        let (b, t, f) = mel.dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(f, 80);
        assert!(t > 0, "expected positive time frames, got {t}");
    }
}
