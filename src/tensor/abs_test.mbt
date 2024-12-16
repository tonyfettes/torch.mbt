test "abs" {
  let a : @tensor.Tensor[Float] = @tensor.tensor(
    [-2.0755, 1.0226, -0.0831, 0.4806],
  )
  inspect!(
    a.abs(),
    content="[2.075500011444092, 1.0226000547409058, 0.08309999853372574, 0.4805999994277954]",
  )
}

test "abs/backward" (it : @test.T) {
  let a : @tensor.Tensor[Float] = @tensor.tensor(
    [-2.0755, 1.0226, -0.0831, 0.4806],
  )
  let b = a.abs()
  b.sum().backward()
  @json.inspect!(a.grad, content=[-1, 1, -1, 1])
  it.writeln(
    #|import torch
    ,
  )
  it.writeln(
    #|def test_abs_backward():
    $|    a = torch.tensor(\{a.to_json().stringify()}, requires_grad=True, dtype=torch.float32)
    $|    b = torch.abs(a)
    $|    b.sum().backward()
    $|    assert torch.allclose(a.grad, torch.tensor(\{a.grad.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  it.snapshot!(filename="tensor_abs_test.py")
}
