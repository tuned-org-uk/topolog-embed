//! This implements ZCA whitening with a numerically stable Newton-Schulz iteration for the matrix inverse square root,
//!  avoiding explicit eigendecomposition while preserving topology without L2 normalization.

use burn::tensor::{ElementConversion, Tensor, backend::Backend};

#[derive(Debug, Clone)]
pub struct WhiteningConfig {
    pub eps: f64,
    pub method: WhiteningMethod,
}

#[derive(Debug, Clone, Copy)]
pub enum WhiteningMethod {
    Zca,
    Pca,
    None,
}

impl Default for WhiteningConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            method: WhiteningMethod::Zca,
        }
    }
}

/// Whitening transform (stateless).
/// Does NOT derive Module because it has no learnable parameters.
#[derive(Debug, Clone)]
pub struct Whitening {
    cfg: WhiteningConfig,
}

impl Whitening {
    pub fn new(cfg: WhiteningConfig) -> Self {
        Self { cfg }
    }

    /// Whitens input tensor (N, D) without L2 normalization.
    pub fn forward<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self.cfg.method {
            WhiteningMethod::None => x,
            WhiteningMethod::Zca => self.zca_whiten(x),
            WhiteningMethod::Pca => self.pca_whiten(x),
        }
    }

    fn zca_whiten<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n, d] = x.dims();
        let device = x.device();

        // Calculate mean: [N, D] -> [D]
        let mean = x.clone().mean_dim(0);

        // Reshape explicitly to [1, D] to ensure broadcasting works safely.
        // This avoids potential ambiguity in unsqueeze_dim with const generics.
        let mean_broadcast = mean.reshape([1, d]);

        // Broadcast subtraction: [N, D] - [1, D]
        let x_centered = x.clone() - mean_broadcast;

        // Covariance: (X^T @ X) / n
        // [D, N] @ [N, D] -> [D, D]
        let cov = x_centered
            .clone()
            .transpose()
            .matmul(x_centered.clone())
            .div_scalar(n as f64);

        let eye = Tensor::<B, 2>::eye(d, &device).mul_scalar(self.cfg.eps);
        let cov_reg = cov + eye;

        let whitening_matrix = self.inverse_sqrt_symmetric(cov_reg);
        x_centered.matmul(whitening_matrix)
    }

    fn pca_whiten<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n, d] = x.dims();
        let device = x.device();

        let mean = x.clone().mean_dim(0);
        // Explicit reshape
        let mean_broadcast = mean.reshape([1, d]);

        let x_centered = x.clone() - mean_broadcast;

        let cov = x_centered
            .clone()
            .transpose()
            .matmul(x_centered.clone())
            .div_scalar(n as f64);

        let eye = Tensor::<B, 2>::eye(d, &device).mul_scalar(self.cfg.eps);
        let cov_reg = cov + eye;

        let whitening_matrix = self.inverse_sqrt_symmetric(cov_reg);
        x_centered.matmul(whitening_matrix)
    }

    fn inverse_sqrt_symmetric<B: Backend>(&self, a: Tensor<B, 2>) -> Tensor<B, 2> {
        let [d, _] = a.dims();
        let device = a.device();

        // Frobenious norm approximation for initialization scaling
        let a_norm = a.clone().powf_scalar(2.0).sum().sqrt();
        // Convert 1-element tensor to scalar f64
        let norm_val = a_norm.into_scalar().elem::<f64>();

        let mut y = Tensor::<B, 2>::eye(d, &device).div_scalar(norm_val);

        // Newton-Schulz iteration
        for _ in 0..5 {
            let y2 = y.clone().matmul(y.clone());
            let ay2 = a.clone().matmul(y2);
            let three_i = Tensor::<B, 2>::eye(d, &device).mul_scalar(3.0);
            y = y.matmul(three_i - ay2).div_scalar(2.0);
        }
        y
    }
}
