test "Uniform/sample" {
  @distributions.seed(b"12345678223456783234567842345678")
  let uniform = @distributions.Uniform::new(0.0, 1.0)
  let sample = uniform.sample([2, 3, 4])
  @json.inspect!(sample.shape, content=[2, 3, 4])
}

test "Uniform/rsample" {
  @distributions.seed(b"12345678223456783234567842345678")
  let lower = @tensor.tensor(0.0, requires_grad=true)
  let upper = @tensor.tensor(1.0, requires_grad=true)
  let uniform = @distributions.Uniform::new(lower, upper)
  let sample = uniform.rsample(())
  @json.inspect!(sample, content=0.35221487283706665)
  sample.backward()
  @json.inspect!(lower.grad, content=[0.6477851271629333])
  @json.inspect!(upper.grad, content=[0.35221487283706665])
}

test "Uniform/entropy" {
  @distributions.seed(b"12345678223456783234567842345678")
  let lower = @tensor.tensor(-3.0, requires_grad=true)
  let upper = @tensor.tensor(5.0, requires_grad=true)
  let uniform = @distributions.Uniform::new(lower, upper)
  let entropy = uniform.entropy()
  @json.inspect!(entropy, content=2.079441547393799)
  entropy.backward()
  @json.inspect!(lower.grad, content=[-0.125])
  @json.inspect!(upper.grad, content=[0.125])
}

test "Uniform/entropy/pytorch" (it : @test.T) {
  let lower = @tensor.tensor(-3.0, requires_grad=true)
  let upper = @tensor.tensor(5.0, requires_grad=true)
  let uniform = @distributions.Uniform::new(lower, upper)
  let entropy = uniform.entropy()
  entropy.backward()
  it.writeln(
    #|import torch
    ,
  )
  it.writeln(
    #|def test_uniform_entropy():
    $|    lower = torch.tensor(\{lower.to_json().stringify()}, requires_grad=True)
    $|    upper = torch.tensor(\{upper.to_json().stringify()}, requires_grad=True)
    $|    uniform = torch.distributions.Uniform(lower, upper)
    $|    entropy = uniform.entropy()
    $|    assert torch.allclose(entropy, torch.tensor(\{entropy.to_json().stringify()}))
    $|    entropy.backward()
    $|    assert torch.allclose(lower.grad, torch.tensor(\{lower.grad.to_json().stringify()}))
    $|    assert torch.allclose(upper.grad, torch.tensor(\{upper.grad.to_json().stringify()}))
  )
  it.snapshot!(filename="distributions_uniform_entropy.py")
}
