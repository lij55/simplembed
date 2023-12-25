use anyhow::{Result};


pub trait Embeddings {
     fn embedding(&self, data: Vec<u8>) -> Result<Vec<f32>>;
}
