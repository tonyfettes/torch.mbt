test "Tanh/consistent" (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let model = @nn.Tanh::new()
  let input = distribution.sample([2, 3, 4, 5])
  let output = model.forward(input)
  it.writeln(
    $|import torch
    $|
    $|def test_tanh():
    $|    input = torch.tensor(\{input.to_json().stringify()}, dtype=torch.float32)
    $|    output = torch.tanh(input)
    $|    assert torch.allclose(output, torch.tensor(\{output.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  it.snapshot!(filename="test_tanh.py")
}
