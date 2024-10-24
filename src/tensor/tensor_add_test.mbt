test "Tensor::op_add/scalar" {
  let a = @tensor.Tensor::new(1.0)
  let b = @tensor.Tensor::new(2.0)
  let c = a + b
  inspect!(c, content="3")
}

test "Tensor::op_add/block" {
  let a = @tensor.Tensor::new([1.0, 2.0, 3.0, 4.0])
  let b = @tensor.Tensor::new([6.0, 7.0, 8.0, 9.0])
  let c = a + b
  inspect!(c, content="[7, 9, 11, 13]")
}

test "Tensor::op_add/multi-block" {
  let a = @tensor.Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
  let b = @tensor.Tensor::new([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
  let c = a + b
  inspect!(c, content="[7, 9, 11, 13, 15, 17, 19, 21]")
}

test "Tensor::op_add/not-aligned" {
  let a = @tensor.Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0])
  let b = @tensor.Tensor::new([6.0, 7.0, 8.0, 9.0, 10.0])
  let c = a + b
  inspect!(c, content="[7, 9, 11, 13, 15]")
}

test "add/consistent" (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let a = distribution.sample([2, 3, 4, 5])
  let b = distribution.sample([2, 3, 4, 5])
  let c = a + b
  it.writeln(
    #|import torch
    #|
    #|def test_add():
    $|    a = torch.tensor(\{a.to_json().stringify()})
    $|    b = torch.tensor(\{b.to_json().stringify()})
    #|    c = a + b
    $|    assert (c == torch.tensor(\{c.to_json().stringify()})).all()
  )
  it.snapshot!(filename="test_add.py")
}
