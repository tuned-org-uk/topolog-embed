#![recursion_limit = "256"]

pub mod backend;
pub mod io;
pub mod topolog;

pub use backend::{AutoBackend, get_device};
pub use topolog::{ParametricUmap, ParametricUmapConfig, Whitening, WhiteningConfig};

use std::sync::Once;
static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("debug");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env).try_init();
    });
}
