test "MSELoss" {
  let input = @tensor.tensor([[1.0, 2.0, 3.0]])
  let target = @tensor.tensor([[4.0, 5.0, 6.0]])
  let criterion = @nn.MSELoss::new()
  inspect!(criterion.forward(input, input), content="0")
  inspect!(criterion.forward(input, target), content="9")
}
