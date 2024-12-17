test "Tensor::stack" {
  let input : FixedArray[@tensor.Tensor[Float]] = [
    @tensor.tensor([0.0, 1.0]),
    @tensor.tensor([1.0, 2.0]),
    @tensor.tensor([2.0, 3.0]),
    @tensor.tensor([3.0, 4.0]),
    @tensor.tensor([4.0, 5.0]),
    @tensor.tensor([5.0, 6.0]),
  ]
  let input = @tensor.stack(input)
  @json.inspect!(input, content=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
}

test "panic Tensor::stack/empty" {
  let input : FixedArray[@tensor.Tensor[Float]] = []
  @tensor.stack(input) |> ignore
}

test "panic Tensor::stack/inconsistent-shape" {
  let input : FixedArray[@tensor.Tensor[Float]] = [
    @tensor.tensor([0.0, 1.0]),
    @tensor.tensor([1.0, 2.0, 3.0]),
  ]
  @tensor.stack(input) |> ignore
}
