use candle::{Result, Tensor};

pub trait CacheLike {
    fn offset(&self) -> usize;
    fn update_and_fetch_kv(&mut self, keys: Tensor, values: Tensor) -> Result<(Tensor, Tensor)>;
    fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor>;
}

/// Builds the depthwise-conv input for one streamed chunk and the history for
/// the next one.
///
/// The input is `padding` frames of history (zeros before the first chunk),
/// then `x`, then `padding` zeros, matching the symmetric padding of the
/// non-streaming path. The returned history is the `padding` frames that end
/// at `committed`: the next chunk starts there, because frames after it are
/// re-encoded.
fn conv_input_and_history(
    history: Option<&Tensor>,
    x: &Tensor,
    padding: usize,
    committed: usize,
) -> Result<(Tensor, Tensor)> {
    let (b, _, d) = x.dims3()?;
    let history = match history {
        Some(h) => h.clone(),
        None => Tensor::zeros((b, padding, d), x.dtype(), x.device())?,
    };
    let with_history = Tensor::cat(&[&history, x], 1)?;
    let next_history = with_history.narrow(1, committed, padding)?;
    Ok((with_history.pad_with_zeros(1, 0, padding)?, next_history))
}

#[derive(Debug, Clone, Default)]
pub struct ConformerCache {
    pub keys: Option<Tensor>,
    pub values: Option<Tensor>,
    pub conv: Option<Tensor>,
    pub offset: usize,
}

impl ConformerCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            conv: None,
            offset: 0,
        }
    }

    pub fn update_and_fetch_kv(
        &mut self,
        keys: Tensor,
        values: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, s, _) = keys.dims4()?;
        if let (Some(k), Some(v)) = (&self.keys, &self.values) {
            let new_k = Tensor::cat(&[k, &keys], 2)?;
            let new_v = Tensor::cat(&[v, &values], 2)?;
            self.keys = Some(new_k);
            self.values = Some(new_v);
        } else {
            self.keys = Some(keys);
            self.values = Some(values);
        }
        self.offset += s;
        Ok((
            self.keys.as_ref().unwrap().clone(),
            self.values.as_ref().unwrap().clone(),
        ))
    }

    pub fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor> {
        if padding == 0 {
            return Ok(x.clone());
        }
        let s = x.dims3()?.1;
        let (input, history) = conv_input_and_history(self.conv.as_ref(), x, padding, s)?;
        self.conv = Some(history);
        Ok(input)
    }
}

impl CacheLike for ConformerCache {
    fn offset(&self) -> usize {
        self.offset
    }

    fn update_and_fetch_kv(&mut self, keys: Tensor, values: Tensor) -> Result<(Tensor, Tensor)> {
        ConformerCache::update_and_fetch_kv(self, keys, values)
    }

    fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor> {
        ConformerCache::update_and_fetch_conv(self, x, padding)
    }
}

#[derive(Debug, Clone)]
pub struct RotatingConformerCache {
    pub keys: Option<Tensor>,
    pub values: Option<Tensor>,
    pub conv: Option<Tensor>,
    pub offset: usize,
    capacity: usize,
    cache_drop_size: usize,
}

impl RotatingConformerCache {
    pub fn new(capacity: usize, cache_drop_size: usize) -> Self {
        Self {
            keys: None,
            values: None,
            conv: None,
            offset: 0,
            capacity,
            cache_drop_size,
        }
    }

    pub fn update_and_fetch_kv(
        &mut self,
        keys: Tensor,
        values: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, s, _) = keys.dims4()?;
        let drop = self.cache_drop_size.min(s);
        let to_cache = s.saturating_sub(drop).min(self.capacity);

        // Return the previous context even if this update evicts part of it.
        let prev_keys = self.keys.clone();
        let prev_values = self.values.clone();

        let new_kv = if to_cache > 0 {
            let start = s - drop - to_cache;
            let k_chunk = keys.narrow(2, start, to_cache)?;
            let v_chunk = values.narrow(2, start, to_cache)?;
            if let (Some(k), Some(v)) = (&self.keys, &self.values) {
                let k_cat = Tensor::cat(&[k, &k_chunk], 2)?;
                let v_cat = Tensor::cat(&[v, &v_chunk], 2)?;
                let k_trim = if k_cat.dims4()?.2 > self.capacity {
                    let start = k_cat.dims4()?.2 - self.capacity;
                    k_cat.narrow(2, start, self.capacity)?
                } else {
                    k_cat
                };
                let v_trim = if v_cat.dims4()?.2 > self.capacity {
                    let start = v_cat.dims4()?.2 - self.capacity;
                    v_cat.narrow(2, start, self.capacity)?
                } else {
                    v_cat
                };
                (k_trim, v_trim)
            } else {
                (k_chunk, v_chunk)
            }
        } else {
            (
                self.keys
                    .clone()
                    .unwrap_or_else(|| keys.narrow(2, 0, 0).unwrap()),
                self.values
                    .clone()
                    .unwrap_or_else(|| values.narrow(2, 0, 0).unwrap()),
            )
        };

        self.keys = Some(new_kv.0);
        self.values = Some(new_kv.1);
        self.offset += to_cache;

        let k_out = if let Some(k) = &prev_keys {
            Tensor::cat(&[k, &keys], 2)?
        } else {
            keys
        };
        let v_out = if let Some(v) = &prev_values {
            Tensor::cat(&[v, &values], 2)?
        } else {
            values
        };

        Ok((k_out, v_out))
    }

    pub fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor> {
        if padding == 0 {
            return Ok(x.clone());
        }
        let s = x.dims3()?.1;
        let (input, history) = conv_input_and_history(
            self.conv.as_ref(),
            x,
            padding,
            s.saturating_sub(self.cache_drop_size),
        )?;
        self.conv = Some(history);
        Ok(input)
    }
}

impl CacheLike for RotatingConformerCache {
    fn offset(&self) -> usize {
        self.offset
    }

    fn update_and_fetch_kv(&mut self, keys: Tensor, values: Tensor) -> Result<(Tensor, Tensor)> {
        RotatingConformerCache::update_and_fetch_kv(self, keys, values)
    }

    fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor> {
        RotatingConformerCache::update_and_fetch_conv(self, x, padding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn frames(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, (1, values.len(), 1), &Device::Cpu).unwrap()
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn conv_cache_left_context_is_previous_chunk() {
        let mut cache = ConformerCache::default();
        let first = cache
            .update_and_fetch_conv(&frames(&[1., 2., 3.]), 2)
            .unwrap();
        assert_eq!(values(&first), [0., 0., 1., 2., 3., 0., 0.]);
        let second = cache.update_and_fetch_conv(&frames(&[4.]), 2).unwrap();
        assert_eq!(values(&second), [2., 3., 4., 0., 0.]);
        let third = cache.update_and_fetch_conv(&frames(&[5., 6.]), 2).unwrap();
        assert_eq!(values(&third), [3., 4., 5., 6., 0., 0.]);
    }

    /// With a drop size, the last `drop` frames of each chunk are encoded
    /// again in the next chunk, so the history must end before them.
    #[test]
    fn rotating_conv_cache_history_ends_at_committed_frames() {
        let mut cache = RotatingConformerCache::new(16, 2);
        let first = cache
            .update_and_fetch_conv(&frames(&[1., 2., 3., 4.]), 1)
            .unwrap();
        assert_eq!(values(&first), [0., 1., 2., 3., 4., 0.]);
        // Frames 3 and 4 were not committed and come again.
        let second = cache
            .update_and_fetch_conv(&frames(&[3., 4., 5.]), 1)
            .unwrap();
        assert_eq!(values(&second), [2., 3., 4., 5., 0.]);
        // Only frame 3 was committed this time; a chunk shorter than the drop
        // size commits nothing and leaves the history alone.
        let third = cache.update_and_fetch_conv(&frames(&[4., 5.]), 1).unwrap();
        assert_eq!(values(&third), [3., 4., 5., 0.]);
        let fourth = cache
            .update_and_fetch_conv(&frames(&[4., 5., 6.]), 1)
            .unwrap();
        assert_eq!(values(&fourth), [3., 4., 5., 6., 0.]);
    }
}
