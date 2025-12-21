use burn::tensor::{Distribution, Tensor};
use log::info;
use topolog_embed::{
    // Adjusted crate name to match previous context
    backend::{AutoBackend, get_device},
    topolog::{ParametricUmap, ParametricUmapConfig, Whitening, WhiteningConfig},
};

fn main() {
    // Initialize the logger to see output.
    // Run with `RUST_LOG=info cargo run --example ...` or defaulting inside code if preferred.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("🚀 Starting Topological Embeddings Pipeline Example");

    // 1. Backend & Device Setup
    let device = get_device();
    // The get_device function prints its own info, so we just log the wrapper message.
    info!("✅ Backend initialized successfully");

    // 2. Data Generation
    let n = 1024;
    let d = 256;
    info!(
        "🎲 Generating random input data: {} samples, {} dimensions",
        n, d
    );

    let x: Tensor<AutoBackend, 2> = Tensor::random([n, d], Distribution::Default, &device);
    info!("   Input tensor shape: {:?}", x.dims());

    // 3. Whitening Transform
    info!("✨ Applying ZCA Whitening...");
    info!("   Whitening decorrelates features while preserving global topology.");

    let whitener = Whitening::new(WhiteningConfig::default());

    // Measure simplified timing if desired, or just log start/end
    let xw = whitener.forward(x);
    info!("   Whitening complete. Output shape: {:?}", xw.dims());

    // 4. Parametric UMAP Model
    let hidden_dim = 512;
    let out_dim = 16;
    info!("🧠 Initializing Parametric UMAP model");
    info!(
        "   Architecture: Input({}) -> Hidden({}) -> Output({})",
        d, hidden_dim, out_dim
    );

    let cfg = ParametricUmapConfig {
        in_dim: d,
        hidden_dim,
        out_dim,
    };
    let model = ParametricUmap::<AutoBackend>::init(&cfg, &device);

    // 5. Forward Pass (Embedding)
    info!("📉 Projecting data to low-dimensional space...");
    let z = model.forward(xw);

    info!("✅ Pipeline finished successfully.");
    info!("   Final Embedding Shape: {:?}", z.dims());

    // Optional: Print a small sample of the embedding
    info!(
        "   First embedding vector sample: {}",
        z.slice([0..1, 0..out_dim])
    );
}
