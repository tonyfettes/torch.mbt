test "Categorical::sample" {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Categorical::of_probs(
    @tensor.tensor([0.25, 0.25, 0.25, 0.25]),
  )
  let sample = distribution.sample([2, 3, 4])
  @json.inspect!(sample, content=[
    [[1, 0, 0, 1], [0, 1, 3, 2], [2, 3, 3, 0]],
    [[2, 1, 3, 1], [0, 2, 2, 3], [0, 2, 0, 1]],
  ])
}

test "Categorical::sample/batch" {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Categorical::of_probs(
    @tensor.tensor([
      [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
      [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
    ]),
  )
  let sample = distribution.sample([2, 3, 4])
  @json.inspect!(sample.shape, content=[2, 3, 4, 2, 2])
  @json.inspect!(sample, content=[
    [
      [[[1, 0], [0, 2]], [[0, 1], [2, 3]], [[3, 1], [3, 2]], [[3, 1], [0, 3]]],
      [[[2, 0], [2, 0]], [[1, 1], [3, 3]], [[1, 2], [1, 1]], [[1, 0], [3, 0]]],
      [[[3, 1], [0, 1]], [[2, 0], [1, 1]], [[0, 3], [3, 1]], [[1, 3], [0, 3]]],
    ],
    [
      [[[0, 0], [0, 0]], [[0, 1], [2, 3]], [[1, 0], [1, 2]], [[1, 1], [1, 0]]],
      [[[1, 3], [2, 2]], [[3, 2], [2, 2]], [[1, 1], [3, 3]], [[3, 2], [0, 0]]],
      [[[0, 1], [0, 1]], [[3, 3], [2, 2]], [[2, 3], [3, 0]], [[2, 2], [1, 3]]],
    ],
  ])
}
