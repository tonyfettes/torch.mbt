test "Normal/sample" {
  @distributions.seed(b"12345678223456783234567842345678")
  let normal = @distributions.Normal::new(0.0, 1.0)
  let sample = normal.sample([2, 3, 4])
  @json.inspect!(sample.shape, content=[2, 3, 4])
}

test "Normal/rsample" {
  @distributions.seed(b"12345678223456783234567842345678")
  let lower : @tensor.Tensor[Float] = @tensor.tensor(0.0, requires_grad=true)
  let upper : @tensor.Tensor[Float] = @tensor.tensor(1.0, requires_grad=true)
  let normal = @distributions.Normal::new(lower, upper)
  let sample = normal.rsample(())
  @json.inspect!(sample, content=1.2142412662506104)
  sample.backward()
  @json.inspect!(lower.grad, content=[1])
  @json.inspect!(upper.grad, content=[1.2142412662506104])
}

test "Normal/entropy" {
  @distributions.seed(b"12345678223456783234567842345678")
  let lower : @tensor.Tensor[Float] = @tensor.tensor(-3.0, requires_grad=true)
  let upper : @tensor.Tensor[Float] = @tensor.tensor(5.0, requires_grad=true)
  let normal = @distributions.Normal::new(lower, upper)
  let entropy = normal.entropy()
  @json.inspect!(entropy, content=3.028376579284668)
  entropy.backward()
  @json.inspect!(lower.grad, content=[0])
  @json.inspect!(upper.grad, content=[0.20000000298023224])
}

test "Normal/entropy/pytorch" (it : @test.T) {
  let lower : @tensor.Tensor[Float] = @tensor.tensor(-3.0, requires_grad=true)
  let upper : @tensor.Tensor[Float] = @tensor.tensor(5.0, requires_grad=true)
  let normal = @distributions.Normal::new(lower, upper)
  let entropy = normal.entropy()
  entropy.backward()
  it.writeln(
    #|import torch
    ,
  )
  it.writeln(
    #|def test_uniform_entropy():
    $|    lower = torch.tensor(\{lower.to_json().stringify()}, requires_grad=True)
    $|    upper = torch.tensor(\{upper.to_json().stringify()}, requires_grad=True)
    $|    normal = torch.distributions.Normal(lower, upper)
    $|    entropy = normal.entropy()
    $|    assert torch.allclose(entropy, torch.tensor(\{entropy.to_json().stringify()}))
    $|    entropy.backward()
    $|    assert torch.allclose(lower.grad, torch.tensor(\{lower.grad.to_json().stringify()}))
    $|    assert torch.allclose(upper.grad, torch.tensor(\{upper.grad.to_json().stringify()}))
    ,
  )
  it.snapshot!(filename="distributions_normal_entropy.py")
}
