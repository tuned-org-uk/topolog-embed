# Topolog Embeddings

Topology-preserving embeddings in Rust, built on top of the Burn 0.18 deep learning framework, with first-class Lance/genegraph-storage support for efficient columnar persistence. This crate focuses on preserving the **roughness** and **smoothness** of the original space (whitening, parametric UMAP) without flattening operations like aggressive normalization.

Features
--------

- Burn 0.18-based models, generic over backends (NdArray / WGPU / CUDA).
- ZCA/PCA whitening transform that decorrelates features without L2 normalizing them.
- Parametric UMAP-style encoder implemented as a Burn module.
- Data loaders that:
    - Load from `.lance` or `.parquet` files via `genegraph-storage`’s `load_dense_from_file`.
    - Load directly from `Vec<Vec<f64>>` for in-memory experimentation.
- Designed for multimodal data (text now, images/audio later).

Installation
------------

In `Cargo.toml`:

```toml
[dependencies]
topolog-embeddings = "0.1"
burn = { version = "0.18", default-features = false, features = ["std", "train"] }
log = "0.4"
env_logger = "0.11"
```

Enable the backend you want (CPU by default).

Usage
-----

### 1. Whitening-only demo

This usage example generates synthetic data, applies ZCA whitening, and logs statistics before and after the transform so you can verify mean-centering and variance normalization.

```rust
use burn::tensor::{Distribution, Tensor};
use log::info;
use topolog_embeddings::{
    backend::{AutoBackend, get_device},
    topolog::{Whitening, WhiteningConfig, WhiteningMethod},
};

fn main() {
    // Initialize logger (RUST_LOG=info cargo run --example 02_whitening_demo)
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    info!("🎨 Whitening Transform Demonstration");
    let device = get_device();

    let n = 2048;
    let d = 128;
    info!("Generating random input: {} samples, {} dims", n, d);

    let x: Tensor<AutoBackend, 2> =
        Tensor::random([n, d], Distribution::Default, &device);
    info!("Input shape: {:?}", x.dims());

    // Pre-whitening global mean
    let mean_before = x.clone().mean_dim(0);
    let mean_scalar = mean_before.clone().mean().into_scalar().elem::<f32>();
    info!("Global mean before whitening: {:.6}", mean_scalar);

    // Apply ZCA whitening
    let whitener = Whitening::new(WhiteningConfig {
        eps: 1e-5,
        method: WhiteningMethod::Zca,
    });

    info!("Applying ZCA whitening…");
    let xw = whitener.forward(x);
    info!("Whitened shape: {:?}", xw.dims());

    // Post-whitening global mean
    let mean_after = xw.clone().mean_dim(0);
    let mean_scalar_after = mean_after.clone().mean().into_scalar().elem::<f32>();
    info!("Global mean after whitening: {:.6}", mean_scalar_after);
}
```

Whitening uses a symmetric inverse square-root of the covariance matrix (via a Newton–Schulz–style iteration) to decorrelate features while keeping the embedding in the original coordinate system, rather than compressing to a latent basis. This matches the intent described in the crate’s whitening module.

### 2. Parametric UMAP-style embedding

This example applies whitening and then feeds the whitened features into a parametric UMAP MLP to obtain a low-dimensional embedding.

```rust
use burn::tensor::{Distribution, Tensor};
use log::info;
use topolog_embeddings::{
    backend::{AutoBackend, get_device},
    topolog::{ParametricUmap, ParametricUmapConfig, Whitening, WhiteningConfig},
};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    info!("🚀 Topological Embedding Pipeline (Whitening + Parametric UMAP)");
    let device = get_device();

    // Synthetic input
    let n = 1024;
    let d = 256;
    info!("Generating input data: {} samples, {} dims", n, d);

    let x: Tensor<AutoBackend, 2> =
        Tensor::random([n, d], Distribution::Default, &device);
    info!("Input shape: {:?}", x.dims());

    // Whitening
    let whitener = Whitening::new(WhiteningConfig::default());
    info!("Applying whitening…");
    let xw = whitener.forward(x);
    info!("Whitened shape: {:?}", xw.dims());

    // Parametric UMAP model
    let cfg = ParametricUmapConfig {
        in_dim: d,
        hidden_dim: 512,
        out_dim: 16,
    };
    let model = ParametricUmap::<AutoBackend>::init(&cfg, &device);
    info!(
        "Parametric UMAP: Input({}) -> Hidden({}) -> Output({})",
        cfg.in_dim, cfg.hidden_dim, cfg.out_dim
    );

    // Low-dimensional embedding
    info!("Projecting to low-dimensional space…");
    let z = model.forward(xw);
    info!("Embedding shape: {:?}", z.dims());
    info!("First embedding row: {}", z.slice([0..1, 0..cfg.out_dim]));
}
```

This layout (whitening → parametric encoder) matches the purpose of the crate: preserve topology in the original feature space and then learn a parametric mapping that retains these structures in a lower-dimensional embedding.

### 3. Loading data from file or Vec

The crate also exposes helpers to get Burn tensors from either `Vec<Vec<f64>>` or from `.lance/.parquet` files via `genegraph-storage`’s `load_dense_from_file` implementation, which supports both Lance and Parquet layouts.[^1]

```rust
use std::path::Path;
use topolog_embeddings::{
    backend::{AutoBackend, get_device},
    data::{load_from_file, load_from_vec},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = get_device();

    // 1) From file (.lance or .parquet)
    let path = Path::new("data/my_dataset.lance");
    let x_file = load_from_file::<AutoBackend>(path, &device).await?;
    println!("Loaded from file: {:?}", x_file.dims());

    // 2) From Vec<Vec<f64>>
    let dense: Vec<Vec<f64>> = vec![
        vec![0.1, 0.4, 0.5],
        vec![0.4, 0.5, 0.2],
        vec![0.03, 0.8, 0.56],
    ];
    let x_vec = load_from_vec::<AutoBackend>(dense, &device)?;
    println!("Loaded from vec: {:?}", x_vec.dims());

    Ok(())
}
```

Internally, the file loader constructs a temporary `LanceStorage` and uses `load_dense_from_file` to support both Lance and Parquet formats, then converts the resulting column-major `DenseMatrix<f64>` into a row-major `Tensor<f32>` used by Burn.
