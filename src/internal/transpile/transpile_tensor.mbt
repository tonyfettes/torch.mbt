pub impl Transpile for @tensor.Tensor with transpile(
  self : @tensor.Tensor,
  transpiler : Transpiler
) {
  let variable = transpiler.allocate()
  transpiler.push_model(
    "\{variable} = torch.tensor(\{self.to_json().stringify()})",
  )
  variable
}
