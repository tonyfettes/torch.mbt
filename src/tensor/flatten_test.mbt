test "flatten" {
  let input : @tensor.Tensor[Float] = @tensor.tensor([
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]],
  ])
  inspect!(input.flatten(), content="[1, 2, 3, 4, 5, 6, 7, 8]")
  inspect!(input.flatten(start=1), content="[[1, 2, 3, 4], [5, 6, 7, 8]]")
}
