use pgrx::prelude::*;
use sentence::bert::Bert;
use util::Embeddings;
use visual::vgg::Vgg;
use visual::load_image224;
use visual::vgg::Which::Vgg16;
use std::sync::{OnceLock, Mutex};

pgrx::pg_module_magic!();

#[pg_extern]
fn pgem_text(c: &str) -> Vec<f32> {
    let b = Bert::from(String::from(String::from("thenlper/gte-small")), String::from(".")).unwrap();
    let i = truncate(c, 500);
    b.embedding(i).unwrap()
}

#[pg_extern]
fn pgem_binary(c: Vec<u8>) -> Vec<f32> {
    // let model = Vgg::new(Vgg16).unwrap();
    let model = get_vgg16().lock().unwrap();

    model.embedding(&c).unwrap()
}

#[pg_extern]
fn pgem_load_as224(path: &str) -> Vec<u8> {
    load_image224(path).unwrap()
}

fn truncate(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        None => s,
        Some((idx, _)) => &s[..idx],
    }
}

fn get_vgg16() -> &'static Mutex<Vgg<'static>> {
    static MODEL: OnceLock<Mutex<Vgg>> = OnceLock::new();
    MODEL.get_or_init(|| Mutex::new(Vgg::new(Vgg16).unwrap()))
}


#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hello_pgem() {
        assert!(true);
    }

}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
