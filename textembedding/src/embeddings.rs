use anyhow::{Error, Result};


pub trait Embeddings {
     fn embedding(&self, text: &str) -> Result<Vec<f32>>;
}


