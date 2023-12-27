/*
request example:

response format:
{
  "data": [
    {
      "index": 0,
      "embedding": [
        0.009904979,
        ...
      ],
      "object": "embedding"
    }
  ],
  "model": "Baichuan-Text-Embedding",
  "object": "list",
  "usage": {
    "prompt_tokens": 7,
    "total_tokens": 7
  }
}
 */

use util::Embeddings;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE, USER_AGENT};
use serde_json::Value;
use std::collections::HashMap;
use anyhow::{Error, Result};



pub struct BaiChuan {
    base_url: String,
    secret: String,
}

impl BaiChuan {
    pub fn new(secret: String) -> BaiChuan {
        BaiChuan {
            base_url: String::from("https://api.baichuan-ai.com/v1/"),
            secret,
        }
    }
}

impl Embeddings<str> for BaiChuan {
     fn embedding(&self, text: &str) -> Result<Vec<f32>> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static("pdbembedding"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_str(format!("Bearer {}", self.secret).as_str()).unwrap(),
        );

        let mut reqmap = HashMap::new();
        reqmap.insert("model", "Baichuan-Text-Embedding");
        reqmap.insert("input", text);
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(format!("{}{}", self.base_url, "embeddings"))
            .headers(headers)
            .json(&reqmap)
            .send()?;
        if resp.status() == 200 {
            let resp = resp.json::<Value>()?;

            match resp["data"][0]["embedding"].clone() {
                Value::Array(v) => Ok(v
                    .iter()
                    .map(|v| {
                        v.as_number()
                            .unwrap_or(&serde_json::Number::from(0))
                            .as_f64()
                            .unwrap_or(0.0) as f32
                    })
                    .collect()),
                _ => Err(Error::msg("failed to extract embeddings from response")),
            }
        } else {
            Err(Error::msg(format!(
                "got response {} from server",
                resp.status()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn baichuan() {
        // get key from env BAICHUAN_SECRET and skip if not exist
        use std::env;
        let mysecret = env::var("BAICHUAN_SECRET").unwrap_or_default();

        assert!(!mysecret.is_empty());

        let bc = BaiChuan::new(mysecret);
        let result = bc.embedding("hello");
        assert!(result.is_ok());

    }
    #[test]
    fn baichuan_unauthorized_request() {
        let bc = BaiChuan::new(String::from("sk"));
        let result = bc.embedding("hello");

        assert!(result.is_err());
        assert_eq!(format!("{}", result.unwrap_err().root_cause()),
                   "got response 401 Unauthorized from server");
    }
}
