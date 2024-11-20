test "sigmoid" {
  inspect!(@tensor.tensor(0.0).sigmoid(), content="0.5")
  inspect!(@tensor.tensor(-1.0).sigmoid(), content="0.2689414322376251")
  inspect!(@tensor.tensor(1.0).sigmoid(), content="0.7310585975646973")
  inspect!(
    @tensor.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]).sigmoid(),
    content="[0.01798621006309986, 0.11920291930437088, 0.2689414322376251, 0.5, 0.7310585975646973, 0.8807970285415649, 0.9820137619972229]",
  )
}

test "sigmoid/consistent" (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let a = distribution.sample([2, 3, 4, 5])
  let c = a.sigmoid()
  it.write(
    #|import torch
    #|
    #|def test_sigmoid():
    $|    a = torch.tensor(\{a.to_json().stringify()})
    #|    c = torch.sigmoid(a)
    $|    assert torch.allclose(c, torch.tensor(\{c.to_json().stringify()}))
    #|
    ,
  )
  it.snapshot!(filename="tensor_sigmoid_test.py")
}
