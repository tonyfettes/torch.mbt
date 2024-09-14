type Model (
  @torch.Linear,
  @torch.ReLU,
  @torch.Linear,
  @torch.ReLU,
  @torch.Linear,
  @torch.Softmax,
) derive(Show, ToJson, @json.FromJson)

pub fn Model::new() -> Model {
  (
    @torch.Linear::new(28 * 28, 196),
    @torch.ReLU::new(),
    @torch.Linear::new(196, 49),
    @torch.ReLU::new(),
    @torch.Linear::new(49, 11),
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
  output
}

pub fn Model::parameters(self : Model) -> Iter[@torch.Tensor] {
  Iter::concat(
    self._.0.parameters(),
    Iter::concat(self._.2.parameters(), self._.4.parameters()),
  )
}
