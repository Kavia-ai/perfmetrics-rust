// Main library file for performance_metrics
pub mod config;
pub mod ffi;
pub mod traits;
mod handlers;
mod plugin;

mod observable;

// Re-export FFI functions and traits at the crate root
pub use config::*;
pub use ffi::*;
pub use traits::*;
