test "Tensor::matmul - 2x3 * 3x2" {
  let a = Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let b = Tensor::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
  let c = a.matmul(b)
  inspect!(c.value, content="[58, 64, 139, 154]")
  inspect!(c.shape, content="[2, 2]")
  inspect!(c.block, content="[2, 1]")
  inspect!(
    c.graph,
    content="MatMul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])",
  )
  inspect!(c.ref, content="0")
  inspect!(c.grad, content="[0, 0, 0, 0]")
  inspect!(c, content="[[58, 64], [139, 154]]")
}

test "Tensor::matmul - 1x2 * 2x2" {
  let x = Tensor::new([1.0, 2.0])
  let w = Tensor::new([[1.0, 2.0], [3.0, 4.0]])
  inspect!(x.upgrade().matmul(w), content="[[7, 10]]")
}

test "Tensor::matmul - 1x2 * 2x2" {
  let x = Tensor::new([2.0, 2.0])
  let w = Tensor::new([[2.0, 2.0], [2.0, 2.0]])
  inspect!(x.upgrade().matmul(w), content="[[8, 8]]")
}
