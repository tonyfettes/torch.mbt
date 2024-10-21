test "Softmax" {
  let input = @tensor.tensor([1.0, 2.0, 3.0])
  let softmax = @nn.Softmax::new()
  let output = softmax.forward(input)
  inspect!(
    output,
    content="[0.09003057317038046, 0.24472847105479767, 0.6652409557748219]",
  )
  let target = @tensor.tensor([1.0, 0.0, 0.0])
  let mut loss = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss + target[i] * output[i]
  }
  inspect!(loss, content="0.09003057317038046")
}
