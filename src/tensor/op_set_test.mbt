test "Tensor::op_set" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
  ])
  tensor[1] = [7.0, 8.0, 9.0]
  inspect!(tensor, content="[[1, 2, 3], [7, 8, 9]]")
  tensor[0] = [0.0, 1.0, 2.0]
  inspect!(tensor, content="[[0, 1, 2], [7, 8, 9]]")
}
