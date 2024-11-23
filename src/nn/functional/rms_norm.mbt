pub fn rms_norm(
  input : @tensor.Tensor,
  normalized_shape : @shape.Shape,
  weight : @tensor.Tensor,
  eps~ : Float = 1.0e-5
) -> @tensor.Tensor {
  let norm_shape = normalized_shape.to_fixed_array()
  let norm_dims = []
  for i in (input.shape.length() - norm_shape.length())..<input.shape.length() {
    norm_dims.push(i)
  }
  let rms = @tensor.mean(input * input, dim=norm_dims, keep_dim=true).sqrt()
  let norm = input / (rms + @tensor.tensor(eps))
  return norm * weight
}
