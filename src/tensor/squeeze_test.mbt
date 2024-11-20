test "@tensor.Tensor::squeeze" {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1, 1)
  let a = distribution.sample([1, 2, 1, 3, 1, 4, 1, 5])
  @json.inspect!(a.squeeze().shape, content=[2, 3, 4, 5])
}

test "@tensor.Tensor::squeeze/dim" {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1, 1)
  let a = distribution.sample([1, 2, 1, 3, 1, 4, 1, 5])
  @json.inspect!(a.squeeze(dim=[0]).shape, content=[2, 1, 3, 1, 4, 1, 5])
  @json.inspect!(a.squeeze(dim=[1, 2]).shape, content=[1, 2, 3, 1, 4, 1, 5])
}
