test "sqrt" {
  let a = @tensor.tensor([-2.0755, 1.0226, 0.0831, 0.4806])
  inspect!(
    a.sqrt(),
    content="[NaN, 1.0112369060516357, 0.2882707118988037, 0.693253219127655]",
  )
}

test "sqrt/backward" {
  let a = @tensor.tensor([-2.0755, 1.0226, 0.0831, 0.4806])
  let b = a.sqrt()
  b.backward()
  inspect!(
    a.grad,
    content="[NaN, 0.49444398283958435, 1.7344807386398315, 0.7212371826171875]",
  )
}

test "sqrt/backward/torch" (it : @test.T) {
  let a = @tensor.tensor([1.0226, 0.0831, 0.4806])
  let b = a.sqrt()
  b.backward()
  it.writeln(
    #|import torch
    ,
  )
  it.writeln(
    #|def test_sqrt_backward():
    $|    a = torch.tensor(\{a.to_json().stringify()}, requires_grad=True, dtype=torch.float32)
    $|    b = torch.sqrt(a)
    $|    b.sum().backward()
    $|    assert torch.allclose(a.grad, torch.tensor(\{a.grad.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  it.snapshot!(filename="tensor_sqrt_test.py")
}
