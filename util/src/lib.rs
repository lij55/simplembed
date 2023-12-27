use anyhow::{Result};


pub trait Embeddings<T: ?Sized> {
    fn embedding(&self, data: &T) -> Result<Vec<f32>>;
}

pub fn dump_vec(v: Vec<f32>) {
    println!(
        "[{}]",
        v
            .into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(",")
    );
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert!(true);
    }
}
