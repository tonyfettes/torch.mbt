test "Tensor::max" {
  let x : @tensor.Tensor[Float] = @tensor.tensor([1.0, 2.0, 3.0, 2.0, 1.0])
  let (max, _) = x.max()
  inspect!(max, content="3")
  max.backward()
  inspect!(x.grad, content="[0, 0, 1, 0, 0]")
}

test "Tensor::max/dim" {
  let a : @tensor.Tensor[Float] = @tensor.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
  )
  let (max, idx) = a.max(dim=0)
  @json.inspect!(max, content=[7, 8, 9])
  @json.inspect!(idx, content=[2, 2, 2])
  let (max, idx) = a.max(dim=1)
  @json.inspect!(max, content=[3, 6, 9])
  @json.inspect!(idx, content=[2, 2, 2])
}

test "Tensor::max/keep_dim" {
  let a : @tensor.Tensor[Float] = @tensor.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
  )
  let (max, idx) = a.max(dim=0, keep_dim=true)
  @json.inspect!(max, content=[[7, 8, 9]])
  @json.inspect!(idx, content=[[2, 2, 2]])
  let (max, idx) = a.max(dim=1, keep_dim=true)
  @json.inspect!(max, content=[[3], [6], [9]])
  @json.inspect!(idx, content=[[2], [2], [2]])
}
