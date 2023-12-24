use textembedding::bert::Bert;
use textembedding::embeddings::Embeddings;
// use textembedding::baichuan::BaiChuan;

fn main() {

    // let bc = BaiChuan::new(String::from("sk-729ab9f7012142d73b1d87995f8e5711"));
    // let result = bc.embedding("hello");
    let b = Bert::from(String::from(String::from("thenlper/gte-small")), String::from(".")).unwrap();
    println!("{}",b.embedding("hello").unwrap().len());
}
