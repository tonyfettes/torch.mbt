type Model (
  @torch.Conv2d,
  @torch.Sigmoid,
  @torch.AvgPool2d,
  @torch.Conv2d,
  @torch.Sigmoid,
  @torch.AvgPool2d,
  @torch.Flatten,
  @torch.Linear,
  @torch.Sigmoid,
  @torch.Linear,
  @torch.Sigmoid,
  @torch.Linear,
  @torch.Softmax,
) derive(Show, ToJson, FromJson)

pub fn Model::new() -> Model {
  (
    @torch.Conv2d::new(1, 6, kernel_size=5, padding=2),
    @torch.Sigmoid::new(),
    @torch.AvgPool2d::new(2, stride=2),
    @torch.Conv2d::new(6, 16, kernel_size=5),
    @torch.Sigmoid::new(),
    @torch.AvgPool2d::new(2, stride=2),
    @torch.Flatten::new(),
    @torch.Linear::new(400, 120),
    @torch.Sigmoid::new(),
    @torch.Linear::new(120, 84),
    @torch.Sigmoid::new(),
    @torch.Linear::new(84, 10),
    @torch.Softmax::new(),
  )
}

pub fn Model::forward(self : Model, input : @torch.Tensor) -> @torch.Tensor {
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
  let output = self._.12.forward(output)
  output
}

pub fn Model::parameters(self : Model) -> Iter[@torch.Tensor] {
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
