test "Tensor::permute" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor([
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
  ])
  @json.inspect!(tensor.shape, content=[2, 3])
  @json.inspect!(tensor.value, content=[0, 1, 2, 3, 4, 5])
  @json.inspect!(tensor, content=[[0, 1, 2], [3, 4, 5]])
  let permuted = tensor.permute([1, 0])
  @json.inspect!(permuted.shape, content=[3, 2])
  @json.inspect!(permuted.value, content=[0, 3, 1, 4, 2, 5])
  @json.inspect!(permuted, content=[[0, 3], [1, 4], [2, 5]])
}
