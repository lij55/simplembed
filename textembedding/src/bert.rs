use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};

use anyhow::{Error as E, Result};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};
use std::path::Path;
use crate::baichuan::BaiChuan;
use crate::embeddings::Embeddings;
pub struct Bert {
    tokenizer:  Tokenizer,
    model: BertModel
}

impl Bert {
    pub fn from(model_name: String, cache_path: String) -> Result<Bert> {
        let (model,tokenizer) = prepare_model(model_name.as_str(), cache_path.as_str())?;
        Ok(Bert{
            model,
            tokenizer
        })
    }
}

impl Embeddings for Bert {
    fn embedding(&self, text: &str) -> Result<Vec<f32>> {
        let device = &Device::Cpu;

        let mut tokenizer = self.tokenizer.clone();

        let mut tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;

        let tokens = tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        //println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        // println!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;


        Ok(embeddings
            .get(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        )

    }
}

fn load_model(
    config: PathBuf,
    tokenizer: PathBuf,
    weights: PathBuf,
) -> Result<(BertModel, Tokenizer)> {
    let config = std::fs::read_to_string(config)?;
    let config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

    let device = &Device::Cpu;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &device)? };

    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

fn prepare_model(model_id: &str, catch_path: &str) -> Result<(BertModel, Tokenizer)> {
    let repo = Repo::new(model_id.to_string(), RepoType::Model);

    let api = ApiBuilder::new()
        .with_cache_dir(PathBuf::from(catch_path))
        .build()?;

    let api = api.repo(repo);

    let config = api.get("config.json")?;
    let tokenizer = api.get("tokenizer.json")?;
    let weights = api.get("model.safetensors")?;

    load_model(config, tokenizer, weights)
}

fn prepare_local_model(path: &str) -> Result<(BertModel, Tokenizer)> {
    let basepath = Path::new(path);

    let config = basepath.join("config.json");
    let tokenizer = basepath.join("tokenizer.json");
    let weights = basepath.join("model.safetensors");

    config.try_exists()?;
    tokenizer.try_exists()?;
    tokenizer.try_exists()?;

    load_model(config, tokenizer, weights)
}

