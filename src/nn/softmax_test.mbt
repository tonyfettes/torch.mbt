test "Softmax" {
  let input : @tensor.Tensor[Float] = @tensor.tensor([1.0, 2.0, 3.0])
  let softmax = @nn.Softmax::new()
  let output = softmax.forward(input)
  inspect!(
    output,
    content="[0.09003057330846786, 0.2447284758090973, 0.6652409434318542]",
  )
  let target : @tensor.Tensor[Float] = @tensor.tensor([1.0, 0.0, 0.0])
  let mut loss : @tensor.Tensor[Float] = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss + target[i] * output[i]
  }
  inspect!(loss, content="0.09003057330846786")
}
