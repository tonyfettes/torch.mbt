///|
pub impl Transpile for @tensor.Tensor[Float] with transpile(
  self : @tensor.Tensor[Float],
  transpiler : Transpiler,
  variable : String
) {
  transpiler.line(
    "\{variable} = torch.tensor(\{self.to_json().stringify()}, dtype=torch.float32)",
  )
}
