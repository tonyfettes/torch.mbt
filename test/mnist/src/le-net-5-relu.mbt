type LeNet5ReLU (
  @torch/nn.Conv2d,
  @torch/nn.ReLU,
  @torch/nn.AvgPool2d,
  @torch/nn.Conv2d,
  @torch/nn.ReLU,
  @torch/nn.AvgPool2d,
  @torch/nn.Flatten,
  @torch/nn.Linear,
  @torch/nn.ReLU,
  @torch/nn.Linear,
  @torch/nn.ReLU,
  @torch/nn.Linear,
) derive(Show, ToJson, FromJson)

pub fn LeNet5ReLU::new() -> LeNet5ReLU {
  (
    @torch/nn.Conv2d::new(1, 6, kernel_size=5, padding=2),
    @torch/nn.ReLU::new(),
    @torch/nn.AvgPool2d::new(2, stride=2),
    @torch/nn.Conv2d::new(6, 16, kernel_size=5),
    @torch/nn.ReLU::new(),
    @torch/nn.AvgPool2d::new(2, stride=2),
    @torch/nn.Flatten::new(),
    @torch/nn.Linear::new(400, 120),
    @torch/nn.ReLU::new(),
    @torch/nn.Linear::new(120, 84),
    @torch/nn.ReLU::new(),
    @torch/nn.Linear::new(84, 10),
  )
}

pub fn LeNet5ReLU::forward(self : LeNet5ReLU, input : @torch.Tensor) -> @torch.Tensor {
  @torch/nn.Module::forward(self._, input)
}

pub fn LeNet5ReLU::parameters(self : LeNet5ReLU) -> Iter[@torch.Tensor] {
  @torch/nn.Module::parameters(self._)
}

pub impl @torch/transpile.Transpile for LeNet5ReLU with transpile(
  self : LeNet5ReLU,
  transpiler : @torch/transpile.Transpiler,
  variable : String
) -> Unit {
  @torch/transpile.Transpile::transpile(self._, transpiler, variable)
}
