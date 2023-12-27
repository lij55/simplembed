use pgrx::prelude::*;
use sentence::bert::Bert;
use util::Embeddings;
use visual::vgg::Vgg;
//use visual::visual::Embeddings;

pgrx::pg_module_magic!();

#[pg_extern]
fn pgem_text(c: &str) -> Vec<f32> {
    let b = Bert::from(String::from(String::from("thenlper/gte-small")), String::from(".")).unwrap();
    let i = truncate(c, 500);
    b.embedding(i).unwrap()

}

#[pg_extern]
fn pgem_binary() -> &'static str {
    "Hello, pgem"
}

fn truncate(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        None => s,
        Some((idx, _)) => &s[..idx],
    }
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
