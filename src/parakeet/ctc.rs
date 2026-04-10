use candle::{D, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder, ops::log_softmax};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ConvASRDecoderArgs {
    pub feat_in: usize,
    pub num_classes: usize,
    pub vocabulary: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AuxCTCArgs {
    pub decoder: ConvASRDecoderArgs,
}

#[derive(Debug, Clone)]
pub struct ConvASRDecoder {
    conv: Conv1d,
    pub num_classes: usize,
}

impl ConvASRDecoder {
    pub fn load(args: &ConvASRDecoderArgs, vb: VarBuilder) -> Result<Self> {
        let num_classes = if args.num_classes > 0 {
            args.num_classes
        } else {
            args.vocabulary.len()
        } + 1;
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let weight = vb.get((num_classes, args.feat_in, 1), "weight")?;
        let bias = vb.get((num_classes,), "bias")?;
        let conv = Conv1d::new(weight, Some(bias), cfg);
        Ok(Self { conv, num_classes })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?; // (B, C, T)
        let logits = self.conv.forward(&x)?;
        let logits = logits.transpose(1, 2)?; // (B, T, C)
        log_softmax(&logits, D::Minus1)
    }
}
