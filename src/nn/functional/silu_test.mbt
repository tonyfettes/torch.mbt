test "silu" {
  let a = @tensor.tensor(1.0)
  @json.inspect!(@functional.silu(a), content=0.7310585975646973)
  let b = @tensor.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  @json.inspect!(
    @functional.silu(b),
    content=[
      [0.7310585975646973, 1.7615940570831299, 2.857722282409668],
      [3.9280550479888916, 4.966536045074463, 5.985164642333984],
    ],
  )
}

test "silu/consistent" (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let a = distribution.sample([2, 3, 4, 5])
  let b = @functional.silu(a)
  it.writeln(
    $|import torch
    $|
    $|def test_silu():
    $|    a = torch.tensor(\{a.to_json().stringify()}, dtype=torch.float32)
    $|    b = torch.nn.functional.silu(a)
    $|    assert torch.allclose(b, torch.tensor(\{b.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  it.snapshot!(filename="tensor_silu_test.py")
}
