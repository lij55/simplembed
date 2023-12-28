#[macro_use]
pub mod vgg;

pub fn load_image224<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<Vec<u8>> {
    let img = image::io::Reader::open(p)?
        .decode()?
        .resize_to_fill(224, 224, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    Ok(img.into_raw())
}