type Simard (
  @torch/nn.Flatten,
  @torch/nn.Linear,
  @torch/nn.ReLU,
  @torch/nn.Linear,
) derive(Show, ToJson, FromJson)

pub fn Simard::new() -> Simard {
  (
    @torch/nn.Flatten::new(),
    @torch/nn.Linear::new(28 * 28, 800),
    @torch/nn.ReLU::new(),
    @torch/nn.Linear::new(800, 10),
  )
}

pub fn Simard::forward(self : Simard, input : @torch.Tensor) -> @torch.Tensor {
  @torch/nn.Module::forward(self._, input)
}

pub fn Simard::parameters(self : Simard) -> Iter[@torch.Tensor] {
  @torch/nn.Module::parameters(self._)
}
