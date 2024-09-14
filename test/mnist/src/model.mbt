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
    @torch.Linear::new(28 * 28, 14 * 14),
    @torch.ReLU::new(),
    @torch.Linear::new(14 * 14, 7 * 7),
    @torch.ReLU::new(),
    @torch.Linear::new(7 * 7, 11),
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

fn Model::parameters(self : Model) -> Iter[@torch.Tensor] {
  Iter::concat(
    self._.0.parameters(),
    Iter::concat(self._.2.parameters(), self._.4.parameters()),
  )
}
