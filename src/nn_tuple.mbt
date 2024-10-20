pub impl[A : Module, B : Module, C : Module] Module for (A, B, C) with forward(
  self : (A, B, C),
  input : Tensor
) -> Tensor {
  input |> forward(self.0) |> forward(self.1) |> forward(self.2)
}

pub impl[A : Module, B : Module, C : Module] Module for (A, B, C) with parameters(
  self : (A, B, C)
) -> Iter[Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
}
