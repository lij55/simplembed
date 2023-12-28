use util::Embeddings;

//use crate::embeddings::Embeddings;
use candle_core::{Device, DType, IndexOp, D, Tensor};
use candle_nn::{ModuleT, VarBuilder};
use candle_transformers::models::vgg::{Models, Vgg as VggModel};
use anyhow::{Error, Result};


pub enum Which {
    Vgg13,
    Vgg16,
    Vgg19,
}

pub struct Vgg<'a> {
    model: VggModel<'a>,
}

impl<'a> Vgg<'a> {
    pub fn new(which: Which) -> Result<Vgg<'a>> {
        let api = hf_hub::api::sync::Api::new()?;
        let device = &Device::Cpu;

        let repo = match which {
            Which::Vgg13 => "timm/vgg13.tv_in1k",
            Which::Vgg16 => "timm/vgg16.tv_in1k",
            Which::Vgg19 => "timm/vgg19.tv_in1k",
        };
        let api = api.model(repo.into());
        let filename = "model.safetensors";
        let model_file = api.get(filename)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = match which {
            Which::Vgg13 => VggModel::new(vb, Models::Vgg13),
            Which::Vgg16 => VggModel::new(vb, Models::Vgg16),
            Which::Vgg19 => VggModel::new(vb, Models::Vgg19),
        };


        match model {
            Ok(v) => Ok(Vgg{model:v}),
            _ => Err(Error::msg("failed to load vgg model")),
        }
    }
}

impl<'a> Embeddings<Vec<u8>> for Vgg<'a>  {

    fn embedding(&self, data: &Vec<u8>) -> anyhow::Result<Vec<f32>> {
        let data = Tensor::from_slice(data, (224, 224, 3), &Device::Cpu)?.permute((2, 0, 1))?;
        let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&[0.229f32, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
        let image = (data.to_dtype(DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?;

        let logits = self.model.forward_t(&image, /*train=*/ false)?;

        let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
            .i(0)?
            .to_vec1::<f32>()?;
        Ok(prs.iter().map(|x| if *x < (0.0001 as f32) {
            0.0
        } else { *x }).collect())
    }
}



#[cfg(test)]
mod tests {
    use crate::vgg::Which::Vgg16;
    use crate::load_image224;
    use super::*;

    #[test]
    fn vgg() {
        let image = load_image224("testdata/postgresql.jpg").unwrap();
        let model = Vgg::new(Vgg16).unwrap();

        let ebd = model.embedding(&image);

        assert!(ebd.is_ok());
        assert_eq!(ebd.unwrap().len(), 1000);

    }

}