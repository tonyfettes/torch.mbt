pub fn uniform(tensor : @tensor.Tensor, lower : Float, upper : Float) -> Unit {
  let distribution = @distributions.Uniform::new(lower, upper)
  distribution.sample_inplace(tensor)
}
