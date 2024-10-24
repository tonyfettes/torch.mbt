pub fn infer[Model : @torch/nn.Module](
  model : Model,
  input : FixedArray[Int]
) -> FixedArray[Float] {
  let input = input.map(fn(value) { value.to_float() / 255.0 })
  let input = @torch.tensor(input).reshape([1, 1, 28, 28])
  model.forward(input).value
}
