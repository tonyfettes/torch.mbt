pub fn normal_(
  tensor : @tensor.Tensor,
  mean~ : Float = 0.0,
  std~ : Float = 1.0
) -> Unit {
  let distribution = @distributions.Normal::new(
    mean=mean.to_double(),
    std=std.to_double(),
  )
  distribution.sample_inplace(tensor)
}
