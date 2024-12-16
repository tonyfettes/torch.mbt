test "panic Tensor::iter - Scalar" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(1.0)
  tensor.iter().collect() |> ignore()
}

test "Tensor::iter - Vector" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor([0.0, 1.0, 2.0, 3.0])
  inspect!(tensor.iter().collect(), content="[0, 1, 2, 3]")
}

test "Tensor::iter - Matrix" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor([[0.0, 1.0], [2.0, 3.0]])
  inspect!(tensor.iter().collect(), content="[[0, 1], [2, 3]]")
}

test "Tensor::iter - Tensor" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(
    [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
  )
  inspect!(
    tensor.iter().collect(),
    content="[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]",
  )
}

test "Tensor::iter/break" {
  let a : @tensor.Tensor[Float] = @tensor.tensor([[1.0, 2.0], [3.0, 4.0]])
  let bs = []
  for b in a {
    bs.push(b)
    break
  }
  @json.inspect!(bs, content=[[1, 2]])
}
