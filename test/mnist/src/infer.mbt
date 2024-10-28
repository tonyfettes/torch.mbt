pub fn infer[Model : @torch/nn.Module](
  model : Model,
  input : FixedArray[Float]
) -> FixedArray[Float] {
  let input = @torch.tensor(input).reshape([1, 1, 28, 28])
  model.forward(input).value()
}
