use image::{imageops::FilterType, GenericImageView};
use ndarray::{
    s, stack, Array1, Array2, Array3, Array4, ArrayBase, ArrayD, ArrayView3, Axis, CowArray, Dim,
    OwnedRepr, ViewRepr,
};

use ort::{inputs, CUDAExecutionProvider, GraphOptimizationLevel, Session, Tensor, Value};
fn main() {
    let model = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .with_model_from_file("visual.onnx")
        .unwrap();

    let image = image::open("1.JPG").unwrap();

    let mean = vec![0.48145466, 0.4578275, 0.40821073]; // CLIP Dataset
    let std = vec![0.26862954, 0.26130258, 0.27577711];

    let outputs = model
        .run(ort::inputs![Array1::from_vec(image.)].unwrap())
        .unwrap();
    let output = outputs["output0"]
        .extract_tensor::<f32>()
        .unwrap()
        .view()
        .t()
        .into_owned();
    println!("{}", output);
}
