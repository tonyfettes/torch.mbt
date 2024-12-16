test "eye_" {
  let a : @tensor.Tensor[Float] = @tensor.tensor([[1, 2], [3, 4]])
  @init.eye_(a)
  @json.inspect!(a, content=[[1, 0], [0, 1]])
}

test "eye_/rectangle" {
  let a : @tensor.Tensor[Float] = @tensor.tensor([[1, 2, 3], [4, 5, 6]])
  @init.eye_(a)
  @json.inspect!(a, content=[[1, 0, 0], [0, 1, 0]])
  let b : @tensor.Tensor[Float] = @tensor.tensor([[1, 2], [3, 4], [5, 6]])
  @init.eye_(b)
  @json.inspect!(b, content=[[1, 0], [0, 1], [0, 0]])
}

test "panic eye_/non-2d" {
  let a : @tensor.Tensor[Float] = @tensor.tensor([1, 2, 3])
  @init.eye_(a)
}
