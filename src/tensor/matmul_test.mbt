test "Tensor::matmul - 2x3 * 3x2" {
  let a : @tensor.Tensor[Float] = @tensor.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
  ])
  let b : @tensor.Tensor[Float] = @tensor.tensor([
    [7.0, 8.0],
    [9.0, 10.0],
    [11.0, 12.0],
  ])
  let c = a.matmul(b)
  inspect!(c.value, content="[58, 64, 139, 154]")
  inspect!(c.shape, content="[2, 2]")
  inspect!(c.block(), content="[2, 1]")
  inspect!(c.grad, content="[0, 0, 0, 0]")
  inspect!(c, content="[[58, 64], [139, 154]]")
}

test "Tensor::matmul - 1x2 * 2x2" {
  let x : @tensor.Tensor[Float] = @tensor.tensor([[1.0, 2.0]])
  let w : @tensor.Tensor[Float] = @tensor.tensor([[1.0, 2.0], [3.0, 4.0]])
  inspect!(x.matmul(w), content="[[7, 10]]")
}

test "Tensor::matmul - 1x2 * 2x2" {
  let x : @tensor.Tensor[Float] = @tensor.tensor([[2.0, 2.0]])
  let w : @tensor.Tensor[Float] = @tensor.tensor([[2.0, 2.0], [2.0, 2.0]])
  inspect!(x.matmul(w), content="[[8, 8]]")
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
  it.writeln(
    #|import torch
    ,
  )
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1024.0, 1024.0)
  // 1x2 @ 2x4
  let a = distribution.sample([1, 2])
  let b = distribution.sample([2, 4])
  let c = a.matmul(b)
  it.writeln(
    #|def test_matmul_1x2_2x4():
    $|    a = torch.tensor(\{a.to_json().stringify()}, dtype=torch.float32)
    $|    b = torch.tensor(\{b.to_json().stringify()}, dtype=torch.float32)
    #|    c = torch.matmul(a, b)
    $|    assert torch.allclose(c, torch.tensor(\{c.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  // 1x8 @ 8x16
  let a = distribution.sample([1, 8])
  let b = distribution.sample([8, 16])
  let c = a.matmul(b)
  it.writeln(
    #|def test_matmul_1x8_8x16():
    $|    a = torch.tensor(\{a.to_json().stringify()}, dtype=torch.float32)
    $|    b = torch.tensor(\{b.to_json().stringify()}, dtype=torch.float32)
    #|    c = torch.matmul(a, b)
    $|    assert torch.allclose(c, torch.tensor(\{c.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  // (2x3 @ 3x2) @ (2x4 @ 4x2)
  let a = distribution.sample([2, 3])
  let b = distribution.sample([3, 2])
  let c = a.matmul(b)
  let d = distribution.sample([2, 4])
  let e = distribution.sample([4, 2])
  let f = d.matmul(e)
  let g = c.matmul(f)
  it.writeln(
    $|def test_matmul_multi_2x3_3x2_2x4_4x2():
    $|    a = torch.tensor(\{a.to_json().stringify()}, dtype=torch.float32)
    $|    b = torch.tensor(\{b.to_json().stringify()}, dtype=torch.float32)
    $|    c = torch.matmul(a, b)
    $|    d = torch.tensor(\{d.to_json().stringify()}, dtype=torch.float32)
    $|    e = torch.tensor(\{e.to_json().stringify()}, dtype=torch.float32)
    $|    f = torch.matmul(d, e)
    $|    g = torch.matmul(c, f)
    $|    assert torch.allclose(g, torch.tensor(\{g.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  it.snapshot!(filename="tensor_matmul_test.py")
}
