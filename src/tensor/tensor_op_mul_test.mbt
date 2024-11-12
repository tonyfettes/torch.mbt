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

test "mul/consistent" (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1024.0, 1024.0)
  let a = distribution.sample([2, 3, 4, 5])
  let b = distribution.sample([2, 3, 4, 5])
  let c = a * b
  it.write(
    #|import torch
    #|
    #|def test_mul():
    $|    a = torch.tensor(\{a.to_json().stringify()})
    $|    b = torch.tensor(\{b.to_json().stringify()})
    #|    c = a * b
    $|    assert torch.allclose(c, torch.tensor(\{c.to_json().stringify()}))
    #|
    ,
  )
  it.snapshot!(filename="test_mul.py")
}
