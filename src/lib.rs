// src/lib.rs — re-exports all modules for integration tests and the binary
pub mod benchmarking;
pub mod engine;
pub mod error;
pub(crate) mod gemm;
pub mod models;
pub mod traits;
pub mod utils;
