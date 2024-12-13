///|
pub fn uniform_(tensor : @tensor.Tensor[Float], lower : Float, upper : Float) -> Unit {
  @distributions.Uniform::new(lower, upper).sample_(tensor)
}
