test "avg_pool2d" {
  let input = @tensor.tensor(
    [
      [
        [
          [1.0, 2.0, 3.0, 4.0],
          [2.0, 3.0, 4.0, 5.0],
          [3.0, 4.0, 5.0, 6.0],
          [4.0, 5.0, 6.0, 7.0],
        ],
      ],
    ],
  )
  inspect!(
    @tensor.avg_pool2d(input, kernel_size=(2, 2), stride=(2, 2)),
    content="[[[[2, 4], [4, 6]]]]",
  )
  inspect!(
    @tensor.avg_pool2d(input, kernel_size=(2, 2), stride=(1, 1)),
    content="[[[[2, 3, 4], [3, 4, 5], [4, 5, 6]]]]",
  )
}

test "avg_pool2d/consistent" (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let a = distribution.sample([2, 3, 4, 5])
  let c = @tensor.avg_pool2d(a, kernel_size=(2, 2), stride=(2, 2))
  it.writeln(
    #|import torch
    #|
    #|def test_avg_pool2d():
    $|    a = torch.tensor(\{a.to_json().stringify()})
    #|    c = torch.nn.functional.avg_pool2d(a, kernel_size=(2, 2), stride=(2, 2))
    $|    assert torch.allclose(c, torch.tensor(\{c.to_json().stringify()}))
    ,
  )
  it.snapshot!(filename="test_avg_pool2d.py")
}
