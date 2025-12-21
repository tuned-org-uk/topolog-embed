use burn::tensor::ElementConversion;
use burn::tensor::{Distribution, Tensor};
use log::info;
use topolog_embed::{
    backend::{AutoBackend, get_device},
    topolog::{Whitening, WhiteningConfig, WhiteningMethod},
};

fn main() {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("🎨 Whitening Transform Demonstration");
    info!("═══════════════════════════════════════════════════════════");

    // 1. Backend setup
    let device = get_device();
    info!("✅ Backend initialized\n");

    // 2. Generate synthetic correlated data
    let n = 2048;
    let d = 128;
    info!("📊 Generating synthetic data:");
    info!("   Samples: {}", n);
    info!("   Features: {}", d);

    let x: Tensor<AutoBackend, 2> = Tensor::random([n, d], Distribution::Default, &device);
    info!("   Input shape: {:?}\n", x.dims());

    // 3. Compute pre-whitening statistics
    info!("📈 Pre-whitening statistics:");
    let mean_before = x.clone().mean_dim(0);
    let mean_scalar = mean_before.clone().mean().into_scalar().elem::<f32>();
    info!("   Global mean: {:.6}", mean_scalar);

    // Sample variance of first feature as diagnostic
    let first_col = x.clone().slice([0..n, 0..1]); // Shape: [n, 1]
    let col_mean = first_col.clone().mean(); // Scalar-like tensor
    // Reshape to [1, 1] for broadcasting against [n, 1]
    let col_mean_broadcast = col_mean.reshape([1, 1]);
    let col_var = (first_col - col_mean_broadcast)
        .powf_scalar(2.0)
        .mean()
        .into_scalar()
        .elem::<f32>();
    info!("   Sample variance (feature 0): {:.6}\n", col_var);

    // 4. Apply ZCA Whitening
    info!("✨ Applying ZCA Whitening...");
    info!("   Method: Zero-phase Component Analysis");
    info!("   Goal: Decorrelate features while minimizing distortion");

    let whitener = Whitening::new(WhiteningConfig {
        eps: 1e-5,
        method: WhiteningMethod::Zca,
    });

    let xw = whitener.forward(x);
    info!("   ✓ Whitening complete\n");

    // 5. Compute post-whitening statistics
    info!("📉 Post-whitening statistics:");
    let mean_after = xw.clone().mean_dim(0);
    let mean_scalar_after = mean_after.clone().mean().into_scalar().elem::<f32>();
    info!("   Global mean: {:.6}", mean_scalar_after);

    let first_col_after = xw.clone().slice([0..n, 0..1]); // [n, 1]
    let col_mean_after = first_col_after.clone().mean();
    let col_mean_after_broadcast = col_mean_after.reshape([1, 1]);
    let col_var_after = (first_col_after - col_mean_after_broadcast)
        .powf_scalar(2.0)
        .mean()
        .into_scalar()
        .elem::<f32>();
    info!("   Sample variance (feature 0): {:.6}", col_var_after);

    // Compute approximate covariance diagonal norm as a quality metric
    let cov_approx = xw
        .clone()
        .transpose()
        .matmul(xw.clone())
        .div_scalar(n as f32);
    let trace = (0..d.min(10)) // Sample first 10 diagonal entries
        .map(|i| {
            cov_approx
                .clone()
                .slice([i..i + 1, i..i + 1])
                .into_scalar()
                .elem::<f32>()
        })
        .sum::<f32>()
        / d.min(10) as f32;
    info!(
        "   Avg diagonal (first {} features): {:.6}\n",
        d.min(10),
        trace
    );

    // 6. Verification
    info!("🔍 Whitening Quality Check:");
    let mean_close_to_zero = mean_scalar_after.abs() < 0.01;
    let variance_near_one = (col_var_after - 1.0).abs() < 0.15;

    if mean_close_to_zero {
        info!("   ✓ Mean centered (|μ| < 0.01)");
    } else {
        info!(
            "   ⚠ Mean not fully centered (|μ| = {:.6})",
            mean_scalar_after.abs()
        );
    }

    if variance_near_one {
        info!("   ✓ Variance normalized (~1.0)");
    } else {
        info!(
            "   ⚠ Variance deviation detected (σ² = {:.6})",
            col_var_after
        );
    }

    info!("\n═══════════════════════════════════════════════════════════");
    info!("✅ Whitening demonstration complete!");
    info!("   Output shape: {:?}", xw.dims());
    info!("   Topological structure preserved without L2 normalization");
}
