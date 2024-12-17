test "Tensor::moveaxis" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor([
    [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
    [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
    [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
    [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
  ])
  @json.inspect!(tensor.shape, content=[4, 2, 3])
  let moved = tensor.moveaxis(-1, 0)
  @json.inspect!(moved.shape, content=[3, 4, 2])
  @json.inspect!(moved, content=[
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    [[1, 1], [1, 1], [1, 1], [1, 1]],
    [[2, 2], [2, 2], [2, 2], [2, 2]],
  ])
}
