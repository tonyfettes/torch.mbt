pub impl Transpile for @tensor.Tensor with transpile(
  self : @tensor.Tensor,
  transpiler : Transpiler,
  variable : String
) {
  transpiler.line(
    "\{variable} = torch.tensor(\{self.to_json().stringify()}, dtype=torch.float32)",
  )
}
