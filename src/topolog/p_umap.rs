//! Simple 2-layer MLP for now; can be extended with batch norm or dropout later.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, backend::Backend},
};

#[derive(Debug, Clone)]
pub struct ParametricUmapConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
}

#[derive(Module, Debug)]
pub struct ParametricUmap<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    act: Relu,
}

impl<B: Backend> ParametricUmap<B> {
    pub fn init(cfg: &ParametricUmapConfig, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(cfg.in_dim, cfg.hidden_dim).init(device),
            l2: LinearConfig::new(cfg.hidden_dim, cfg.out_dim).init(device),
            act: Relu::new(),
        }
    }

    /// Forward: (N, in_dim) -> (N, out_dim)
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.l1.forward(x);
        let x = self.act.forward(x);
        self.l2.forward(x)
    }
}
