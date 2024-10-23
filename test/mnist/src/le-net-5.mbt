type LeNet5 (
  @torch/nn.Conv2d,
  @torch/nn.Sigmoid,
  @torch/nn.AvgPool2d,
  @torch/nn.Conv2d,
  @torch/nn.Sigmoid,
  @torch/nn.AvgPool2d,
  @torch/nn.Flatten,
  @torch/nn.Linear,
  @torch/nn.Sigmoid,
  @torch/nn.Linear,
  @torch/nn.Sigmoid,
  @torch/nn.Linear,
) derive(Show, ToJson, FromJson)

pub fn LeNet5::new() -> LeNet5 {
  (
    @torch/nn.Conv2d::new(1, 6, kernel_size=5, padding=2),
    @torch/nn.Sigmoid::new(),
    @torch/nn.AvgPool2d::new(2, stride=2),
    @torch/nn.Conv2d::new(6, 16, kernel_size=5),
    @torch/nn.Sigmoid::new(),
    @torch/nn.AvgPool2d::new(2, stride=2),
    @torch/nn.Flatten::new(),
    @torch/nn.Linear::new(400, 120),
    @torch/nn.Sigmoid::new(),
    @torch/nn.Linear::new(120, 84),
    @torch/nn.Sigmoid::new(),
    @torch/nn.Linear::new(84, 10),
  )
}

pub fn LeNet5::forward(self : LeNet5, input : @torch.Tensor) -> @torch.Tensor {
  @torch/nn.Module::forward(self._, input)
}

pub fn LeNet5::parameters(self : LeNet5) -> Iter[@torch.Tensor] {
  @torch/nn.Module::parameters(self._)
}
