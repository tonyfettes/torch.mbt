test "Tensor::op_mul" {
  let a = @tensor.Tensor::new([1.0, 2.0, 3.0])
  let b = @tensor.Tensor::new([4.0, 5.0, 6.0])
  let result = a * b
  inspect!(result, content="[4, 10, 18]")
}

test "Tensor::op_mul - negative" {
  let a = @tensor.Tensor::new([1.0, 2.0, 3.0])
  let b = @tensor.Tensor::new([-4.0, -5.0, -6.0])
  let result = a * b
  inspect!(result, content="[-4, -10, -18]")
}
