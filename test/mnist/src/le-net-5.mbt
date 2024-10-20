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

pub fn LeNet5::forward(
  self : LeNet5,
  input : @torch.Tensor
) -> @torch.Tensor {
  let output = self._.0.forward(input)
  let output = self._.1.forward(output)
  let output = self._.2.forward(output)
  let output = self._.3.forward(output)
  let output = self._.4.forward(output)
  let output = self._.5.forward(output)
  let output = self._.6.forward(output)
  let output = self._.7.forward(output)
  let output = self._.8.forward(output)
  let output = self._.9.forward(output)
  let output = self._.10.forward(output)
  let output = self._.11.forward(output)
  output
}

pub fn LeNet5::parameters(self : LeNet5) -> Iter[@torch.Tensor] {
  Iter::concat(
    self._.0.parameters(),
    Iter::concat(
      self._.3.parameters(),
      Iter::concat(
        self._.7.parameters(),
        Iter::concat(self._.9.parameters(), self._.11.parameters()),
      ),
    ),
  )
}
