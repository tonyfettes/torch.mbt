test "Tensor::broadcast_to" {
  let a = @tensor.zeros([3, 1, 1])
  let b = a.broadcast_to([5, 3, 4, 1])
  @json.inspect!(b.shape, content=[5, 3, 4, 1])
}

test "panic Tensor::broadcast_to/narrow" {
  let a = @tensor.zeros([3, 1, 1])
  a.broadcast_to([5, 1, 4, 1]) |> ignore()
}

test "panic Tensor::broadcast_to/mismatch" {
  let a = @tensor.zeros([3, 1, 1])
  a.broadcast_to([5, 2, 4, 1]) |> ignore()
}

test "Tensor::broadcast_to/scalar" {
  let a = @tensor.tensor(1.0)
  let b = a.broadcast_to([3, 1, 7])
  @json.inspect!(b.shape, content=[3, 1, 7])
  @json.inspect!(
    b,
    content=[
      [[1, 1, 1, 1, 1, 1, 1]],
      [[1, 1, 1, 1, 1, 1, 1]],
      [[1, 1, 1, 1, 1, 1, 1]],
    ],
  )
}

test "Tensor::broadcast_to/vector" {
  let a = @tensor.tensor([1.0, 2.0, 3.0])
  let b = a.broadcast_to([7, 1, 3])
  @json.inspect!(b.shape, content=[7, 1, 3])
  @json.inspect!(
    b,
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

test "Tensor::broadcast_to/matrix" {
  let a = @tensor.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let b = a.broadcast_to([7, 2, 3])
  @json.inspect!(b.shape, content=[7, 2, 3])
  @json.inspect!(
    b,
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
