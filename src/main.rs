// use textembedding::bert::Bert;
// use textembedding::embeddings::Embeddings;
// // use textembedding::baichuan::BaiChuan;
//
// fn main() {
//
//     // let bc = BaiChuan::new(String::from("sk-729ab9f7012142d73b1d87995f8e5711"));
//     // let result = bc.embedding("hello");
//     let b = Bert::from(String::from(String::from("thenlper/gte-small")), String::from(".")).unwrap();
//     println!("{}",b.embedding("hello").unwrap().len());
// }

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{Device, DType, IndexOp, D, Tensor, Result, Error};
use candle_nn::{ModuleT, VarBuilder};
use candle_transformers::models::vgg::{Models, Vgg};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Vgg13,
    Vgg16,
    Vgg19,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Variant of the model to use.
    #[arg(value_enum, long, default_value_t = Which::Vgg16)]
    which: Which,
}

pub fn load_image224<P: AsRef<std::path::Path>>(p: P) -> Result<Tensor> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(Error::wrap)?
        .resize_to_fill(224, 224, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (224, 224, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229f32, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
    (data.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = &Device::Cpu;
    let image = load_image224(args.image)?;

    //println!("loaded image {image:?}");

    let api = hf_hub::api::sync::Api::new()?;
    let repo = match args.which {
        Which::Vgg13 => "timm/vgg13.tv_in1k",
        Which::Vgg16 => "timm/vgg16.tv_in1k",
        Which::Vgg19 => "timm/vgg19.tv_in1k",
    };
    let api = api.model(repo.into());
    let filename = "model.safetensors";
    let model_file = api.get(filename)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = match args.which {
        Which::Vgg13 => Vgg::new(vb, Models::Vgg13)?,
        Which::Vgg16 => Vgg::new(vb, Models::Vgg16)?,
        Which::Vgg19 => Vgg::new(vb, Models::Vgg19)?,
    };
    let logits = model.forward_t(&image, /*train=*/ false)?;

    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;
    let v : Vec<f32> = prs.iter().map(|x| if *x < (0.0001 as f32) {
        0.0
    } else { *x }).collect();
    println!("{:?}", v);

    // Sort the predictions and take the top 5
    // let mut top: Vec<_> = prs.iter().enumerate().collect();
    // top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    // let top = top.into_iter().take(5).collect::<Vec<_>>();
    //
    // // Print the top predictions
    // for &(i, p) in &top {
    //     println!(
    //         "{:50}: {:.2}%",
    //         candle_examples::imagenet::CLASSES[i],
    //         p * 100.0
    //     );
    // }

    Ok(())
}