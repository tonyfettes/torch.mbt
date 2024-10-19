type Simard (
  @torch.Flatten,
  @torch.Linear,
  @torch.ReLU,
  @torch.Linear,
) derive(Show, ToJson, FromJson)

pub fn Simard::new() -> Simard {
  (
    @torch.Flatten::new(),
    @torch.Linear::new(28 * 28, 800),
    @torch.ReLU::new(),
    @torch.Linear::new(800, 10)
  )
}

pub fn Simard::forward(self : Simard, input : @torch.Tensor) -> @torch.Tensor {
  let output = self._.0.forward(input)
  let output = self._.1.forward(output)
  let output = self._.2.forward(output)
  let output = self._.3.forward(output)
  output
}

pub fn Simard::parameters(self : Simard) -> Iter[@torch.Tensor] {
  Iter::concat(
    self._.0.parameters(),
    Iter::concat(self._.1.parameters(), self._.3.parameters()),
  )
}
