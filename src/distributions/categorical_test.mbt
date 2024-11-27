test "Categorical::sample" {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Categorical::of_probs(
    @tensor.tensor([0.25, 0.25, 0.25, 0.25]),
  )
  let sample = distribution.sample([2, 3, 4])
  @json.inspect!(
    sample,
    content=[
      [[1, 0, 0, 1], [0, 1, 3, 2], [2, 3, 3, 0]],
      [[2, 1, 3, 1], [0, 2, 2, 3], [0, 2, 0, 1]],
    ],
  )
}
