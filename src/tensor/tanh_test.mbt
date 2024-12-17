/// See: https://pytorch.org/docs/stable/generated/torch.tanh.html
test "tanh" {
  let a : @tensor.Tensor[Float] = @tensor.tensor([
    0.8986, -0.7279, 1.1745, 0.2611,
  ])
  @json.inspect!(a.tanh(), content=[
    0.715615451335907, -0.6217789053916931, 0.825709342956543, 0.255324125289917,
  ])
}

test "tanh/consistent" (it : @test.T) {
  it.writeln(
    #|import torch
    #|
    ,
  )
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let a = distribution.sample([2, 3, 4, 5])
  let b = a.tanh()
  it.writeln(
    #|def test_tanh():
    $|    a = torch.tensor(\{a.to_json().stringify()}, dtype=torch.float32)
    #|    b = torch.tanh(a)
    $|    assert torch.allclose(b, torch.tensor(\{b.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  it.snapshot!(filename="tensor_tanh_test.py")
}
