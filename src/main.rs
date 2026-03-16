// src/main.rs

use std::sync::OnceLock;

use lambda_http::{run, service_fn, Body, Error, Request, Response};
use serde::{Deserialize, Serialize};
use tracing::info;

use ltembed::engine::ZeroVecEngine;
use ltembed::error::LTEmbedError;
use ltembed::traits::pooling::MeanPooling;

#[derive(Deserialize)]
struct EmbedRequest {
    inputs: Vec<String>,
}

#[derive(Serialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

// Stores Ok(engine) or Err(message) after the first initialization attempt.
static ENGINE: OnceLock<Result<ZeroVecEngine, String>> = OnceLock::new();

fn get_engine() -> Result<&'static ZeroVecEngine, LTEmbedError> {
    ENGINE
        .get_or_init(|| {
            (|| -> Result<ZeroVecEngine, LTEmbedError> {
                let config_str = std::fs::read_to_string("assets/config.json")?;
                ZeroVecEngine::new(
                    "assets/model.safetensors",
                    &config_str,
                    "assets/tokenizer.json",
                    Box::new(MeanPooling),
                )
            })()
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| LTEmbedError::ModelLoad(e.clone()))
}

async fn handler(event: Request) -> Result<Response<Body>, Error> {
    let body_bytes = match event.body() {
        Body::Text(s) => s.as_bytes().to_vec(),
        Body::Binary(b) => b.clone(),
        Body::Empty => Vec::new(),
    };

    let req: EmbedRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return Ok(Response::builder()
                .status(400)
                .header("content-type", "application/json")
                .body(Body::Text(format!(r#"{{"error": "Invalid JSON: {e}"}}"#)))?);
        }
    };

    if req.inputs.is_empty() {
        return Ok(Response::builder()
            .status(400)
            .header("content-type", "application/json")
            .body(Body::Text(r#"{"error": "inputs must not be empty"}"#.to_string()))?);
    }

    let engine = match get_engine() {
        Ok(e) => e,
        Err(e) => {
            return Ok(Response::builder()
                .status(500)
                .header("content-type", "application/json")
                .body(Body::Text(format!(r#"{{"error": "Engine init failed: {e}"}}"#)))?);
        }
    };

    let text_refs: Vec<&str> = req.inputs.iter().map(String::as_str).collect();
    let embeddings = match engine.embed_batch(&text_refs) {
        Ok(v) => v,
        Err(LTEmbedError::InputTooLong { tokens, max }) => {
            // Client error: the library explicitly rejects overlong inputs.
            // Map to 400 so the caller knows to chunk/truncate their text.
            return Ok(Response::builder()
                .status(400)
                .header("content-type", "application/json")
                .body(Body::Text(format!(
                    r#"{{"error": "Input too long: {tokens} tokens exceeds the {max} token limit"}}"#
                )))?);
        }
        Err(e) => {
            return Ok(Response::builder()
                .status(500)
                .header("content-type", "application/json")
                .body(Body::Text(format!(r#"{{"error": "Inference failed: {e}"}}"#)))?);
        }
    };

    let json = serde_json::to_string(&EmbedResponse { embeddings })?;
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(Body::Text(json))?)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    info!("LTEmbed cold start complete");
    run(service_fn(handler)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambda_http::http::Method;

    fn make_request(body: &str) -> Request {
        lambda_http::http::Request::builder()
            .method(Method::POST)
            .uri("/embed")
            .header("content-type", "application/json")
            .body(Body::Text(body.to_string()))
            .unwrap()
            .into()
    }

    #[tokio::test]
    async fn test_malformed_json_returns_400() {
        let req = make_request("not json {{{{");
        let resp = handler(req).await.unwrap();
        assert_eq!(resp.status(), 400);
        let body = match resp.body() {
            Body::Text(s) => s.clone(),
            _ => panic!("expected text body"),
        };
        assert!(body.contains("Invalid JSON"));
    }

    #[tokio::test]
    async fn test_empty_inputs_returns_400() {
        let req = make_request(r#"{"inputs": []}"#);
        let resp = handler(req).await.unwrap();
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn test_overlong_input_returns_400() {
        // The library returns InputTooLong; the handler must map it to 400, not 500.
        // This test requires model assets to reach the embed_batch call.
        if !std::path::Path::new("assets/model.safetensors").exists() {
            eprintln!("Skipping: model assets not found");
            return;
        }
        let long_text = "hello world ".repeat(5000);
        let body = format!(r#"{{"inputs": ["{long_text}"]}}"#);
        let req = make_request(&body);
        let resp = handler(req).await.unwrap();
        assert_eq!(resp.status(), 400);
        let resp_body = match resp.body() {
            Body::Text(s) => s.clone(),
            _ => panic!("expected text body"),
        };
        assert!(resp_body.contains("too long") || resp_body.contains("token"));
    }

    #[tokio::test]
    async fn test_valid_request_returns_200_with_embeddings() {
        if !std::path::Path::new("assets/model.safetensors").exists() {
            eprintln!("Skipping: model assets not found");
            return;
        }
        let req = make_request(r#"{"inputs": ["query: Hello, world!"]}"#);
        let resp = handler(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let body = match resp.body() {
            Body::Text(s) => s.clone(),
            _ => panic!("expected text body"),
        };
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        let embeddings = parsed["embeddings"].as_array().unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].as_array().unwrap().len(), 384);
    }
}
