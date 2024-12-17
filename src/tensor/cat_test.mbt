test "cat" {
  let tensors : FixedArray[@tensor.Tensor[Float]] = [
    @tensor.tensor([[1.0, 2.0], [3.0, 4.0]]),
    @tensor.tensor([[5.0, 6.0], [7.0, 8.0]]),
  ]
  @json.inspect!(@tensor.cat(tensors, dim=0), content=[
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ])
  @json.inspect!(@tensor.cat(tensors, dim=1), content=[
    [1, 2, 5, 6],
    [3, 4, 7, 8],
  ])
}

test "panic cat/empty" {
  let tensors : FixedArray[@tensor.Tensor[Float]] = []
  @tensor.cat(tensors) |> ignore
}
