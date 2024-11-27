test "Categorical::sample" {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Categorical::of_probs(
    @tensor.tensor([0.25, 0.25, 0.25, 0.25]),
  )
  let sample = distribution.sample([1, 2, 3])
  @json.inspect!(sample, content=0)
}
