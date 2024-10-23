test "Tensor::max" {
  let x = @tensor.tensor([1.0, 2.0, 3.0, 2.0, 1.0])
  let max = x.max()
  inspect!(max, content="3")
  max.backward()
  inspect!(x.grad, content="[0, 0, 1, 0, 0]")
}
