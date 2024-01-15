//use sentence::bert::Bert;
use util::dump_vec;
use util::Embeddings;
// use sentence::baichuan::BaiChuan;
use std::fs;
use std::fs::read_to_string;
use std::io::Write;

use candle_core::{DType, Device, Error, IndexOp, Result, Tensor, D};
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
    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    from: Option<String>,

    #[arg(long)]
    output: Option<String>,

    text: Vec<String>,
}

fn truncate(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        None => s,
        Some((idx, _)) => &s[..idx],
    }
}

fn main() {
    // let args = Args::parse();
    // let b = Bert::from(
    //     String::from(String::from("thenlper/gte-small")),
    //     String::from("."),
    // )
    // .unwrap();
    //
    // if args.from.is_some() {
    //     let mut file = fs::OpenOptions::new()
    //         .append(true)
    //         .create(true)
    //         .open(args.output.clone().unwrap())
    //         .unwrap();
    //     for line in read_to_string(args.from.unwrap()).unwrap().lines() {
    //         let i = truncate(line, 500);
    //         let v: Vec<f32> = match b.embedding(i) {
    //             Ok(v) => v,
    //             _ => continue,
    //         };
    //         file.write_all(
    //             format!(
    //                 "[{}]\n",
    //                 v.into_iter()
    //                     .map(|x| x.to_string())
    //                     .collect::<Vec<String>>()
    //                     .join(",")
    //             )
    //             .as_bytes(),
    //         )
    //         .unwrap();
    //     }
    // } else {
    //     for i in args.text {
    //         let i = truncate(i.as_str(), 500);
    //         let v = b.embedding(i).unwrap();
    //         dump_vec(v);
    //     }
    // }

    // let bc = BaiChuan::new(String::from("sk-729ab9f7012142d73b1d87995f8e5711"));
    // let result = bc.embedding("hello");
}
