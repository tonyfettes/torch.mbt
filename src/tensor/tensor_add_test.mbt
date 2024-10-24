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

test (it : @test.T) {
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let a = distribution.sample([2, 3, 4, 5])
  let b = distribution.sample([2, 3, 4, 5])
  let c = a + b
  it.write(
    #|import torch
    #|
    #|def test_add_random():
    $|    a = torch.tensor(\{a})
    $|    b = torch.tensor(\{b})
    #|    c = a + b
    $|    assert (c == torch.tensor(\{c})).all()
    #|
  )
  it.snapshot!(filename="test_add_random.py")
}
