pub fn normal_(
  tensor : @tensor.Tensor,
  mean~ : Float = 0.0,
  std~ : Float = 1.0
) -> Unit {
  @distributions.Normal::new(mean, std).sample_(tensor)
}
