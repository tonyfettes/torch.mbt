test "Tensor::broadcast - same shape" {
  let x = @tensor.zeros([5, 7, 3])
  let y = @tensor.zeros([5, 7, 3])
  let (bx, by) = @tensor.broadcast(x, y)
  @json.inspect!(bx.shape, content=[5, 7, 3])
  @json.inspect!(by.shape, content=[5, 7, 3])
}

test "Tensor::broadcast - scalar" {
  let x = @tensor.tensor(1.0)
  let y = @tensor.zeros([3, 1, 7])
  let (bx, _) = @tensor.broadcast(x, y)
  @json.inspect!(bx.shape, content=[3, 1, 7])
  @json.inspect!(
    bx,
    content=[
      [[1, 1, 1, 1, 1, 1, 1]],
      [[1, 1, 1, 1, 1, 1, 1]],
      [[1, 1, 1, 1, 1, 1, 1]],
    ],
  )
}

test "Tensor::broadcast - vector" {
  let x = @tensor.tensor([1.0, 2.0, 3.0])
  let y = @tensor.zeros([7, 1, 3])
  let (bx, _) = @tensor.broadcast(x, y)
  @json.inspect!(bx.shape, content=[7, 1, 3])
  @json.inspect!(
    bx,
    content=[
      [[1, 2, 3]],
      [[1, 2, 3]],
      [[1, 2, 3]],
      [[1, 2, 3]],
      [[1, 2, 3]],
      [[1, 2, 3]],
      [[1, 2, 3]],
    ],
  )
}

test "Tensor::broadcast - matrix" {
  let x = @tensor.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let y = @tensor.zeros([7, 2, 3])
  let (bx, _) = @tensor.broadcast(x, y)
  @json.inspect!(bx.shape, content=[7, 2, 3])
  @json.inspect!(
    bx,
    content=[
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
    ],
  )
}

test "Tensor::broadcast" {
  let x = @tensor.zeros([5, 3, 4, 1])
  let y = @tensor.zeros([3, 1, 1])
  let (bx, by) = @tensor.broadcast(x, y)
  @json.inspect!(bx.shape, content=[5, 3, 4, 1])
  @json.inspect!(by.shape, content=[5, 3, 4, 1])
  let x = @tensor.zeros([5, 1, 4, 1])
  let y = @tensor.zeros([3, 1, 1])
  let (bx, by) = @tensor.broadcast(x, y)
  @json.inspect!(bx.shape, content=[5, 3, 4, 1])
  @json.inspect!(by.shape, content=[5, 3, 4, 1])
}

test "panic Tensor::broadcast - shapes mismatch" {
  let x = @tensor.zeros([5, 2, 4, 1])
  let y = @tensor.zeros([3, 1, 1])
  let (bx, by) = @tensor.broadcast(x, y)
  @json.inspect!(bx.shape)
  @json.inspect!(by.shape)
}
