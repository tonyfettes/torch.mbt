test "Tensor::stack" {
  let input : FixedArray[_] = [
    [[@tensor.tensor(0.0), @tensor.tensor(1.0), @tensor.tensor(2.0)]],
    [[@tensor.tensor(3.0), @tensor.tensor(4.0), @tensor.tensor(5.0)]],
  ]
  let input = @tensor.stack(input)
  inspect!(input, content="[[[0, 1, 2]], [[3, 4, 5]]]")
}
