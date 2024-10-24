test "Tensor::matmul - 2x3 * 3x2" {
  let a = @tensor.Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let b = @tensor.Tensor::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
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
  let x = @tensor.Tensor::new([1.0, 2.0])
  let w = @tensor.Tensor::new([[1.0, 2.0], [3.0, 4.0]])
  inspect!(x.upgrade().matmul(w), content="[[7, 10]]")
}

test "Tensor::matmul - 1x2 * 2x2" {
  let x = @tensor.Tensor::new([2.0, 2.0])
  let w = @tensor.Tensor::new([[2.0, 2.0], [2.0, 2.0]])
  inspect!(x.upgrade().matmul(w), content="[[8, 8]]")
}

test "matmul/8@8x16" {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let x = distribution.sample([1, 8])
  let w = distribution.sample([8, 16])
  let y = x.matmul(w)
  inspect!(y.shape, content="[1, 16]")
}

test "matmul/consistent" (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let a = distribution.sample([1, 8])
  let b = distribution.sample([8, 16])
  let c = a.matmul(b)
  it.write(
    #|import torch
    #|
    #|def test_matmul():
    $|    a = torch.tensor(\{a.to_json().stringify()})
    $|    b = torch.tensor(\{b.to_json().stringify()})
    #|    c = torch.matmul(a, b)
    $|    assert torch.allclose(c, torch.tensor(\{c.to_json().stringify()}))
    #|
  )
  it.snapshot!(filename="test_matmul.py")
}