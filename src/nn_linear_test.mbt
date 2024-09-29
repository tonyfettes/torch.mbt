test "Linear" {
  let linear = @torch.Linear::new(2, 2)
  @torch.with_no_grad(
    fn() {
      linear.weight[0][0] = 1.0
      linear.weight[0][1] = 3.0
      linear.weight[1][0] = 2.0
      linear.weight[1][1] = 4.0
      linear.bias[0] = 5.0
      linear.bias[1] = 6.0
    },
  )
  let input = @torch.tensor([1.0, 2.0])
  let optimizer = @torch.SGD::new(linear.parameters(), learning_rate=0.1)
  let output = linear.forward(input)
  inspect!(output, content="[10, 17]")
  let target = @torch.tensor([16.0, 16.0])
  let mut loss = @torch.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss += output[i] - target[i]
  }
  inspect!(loss, content="59")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  inspect!(linear.weight, content="[[0.9, 2.9], [1.8, 3.8]]")
  inspect!(linear.bias, content="[4.9, 5.9]")
  let output = linear.forward(input)
  inspect!(output, content="[9.4, 16.4]")
}

test "Linear with ReLU" {
  let linear = @torch.Linear::new(2, 2)
  @torch.with_no_grad(
    fn() {
      linear.weight[0][0] = 1.0
      linear.weight[0][1] = 3.0
      linear.weight[1][0] = 2.0
      linear.weight[1][1] = 4.0
      linear.bias[0] = 5.0
      linear.bias[1] = 6.0
    },
  )
  let layers = @torch.Sequential::new([linear, @torch.ReLU::new()])
  let optimizer = @torch.SGD::new(linear.parameters(), learning_rate=1)
  let input = @torch.tensor([2.0, 2.0])
  let output = layers.forward(input)
  inspect!(output, content="[11, 20]")
  let target = @torch.tensor([1.0, 0.0])
  let mut loss = @torch.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i].to_tensor() * output[i].log()
  }
  inspect!(loss, content="-2.3978952727983707")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  inspect!(
    linear.weight,
    content="[[1.1818181818181819, 3], [2.1818181818181817, 4]]",
  )
  let output = layers.forward(input)
  inspect!(output, content="[11.818181818181817, 20]")
}

test "Linear with Softmax" {
  let linear = @torch.Linear::new(2, 2)
  @torch.with_no_grad(
    fn() {
      linear.weight[0][0] = 2.0
      linear.weight[0][1] = 2.0
      linear.weight[1][0] = 2.0
      linear.weight[1][1] = 2.0
      linear.bias[0] = 1.0
      linear.bias[1] = 2.0
    },
  )
  let layers = @torch.Sequential::new([linear, @torch.Softmax::new()])
  let optimizer = @torch.SGD::new(linear.parameters(), learning_rate=1)
  let input = @torch.tensor([2.0, 2.0])
  let output = layers.forward(input)
  inspect!(output, content="[0.2689414213699951, 0.7310585786300049]")
  let target = @torch.tensor([1.0, 0.0])
  let mut loss = @torch.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i].to_tensor() * output[i].log()
  }
  inspect!(loss, content="1.3132616875182228")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  let output = layers.forward(input)
  inspect!(output, content="[0.9999947593218812, 0.00000524067811873532]")
}
