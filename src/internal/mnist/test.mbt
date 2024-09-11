test {
  let model = [
      @torch.Linear::new(28 * 28, 128),
      @torch.ReLU::new(),
      @torch.Linear::new(128, 64),
      @torch.ReLU::new(),
      @torch.Linear::new(64, 10),
      @torch.Softmax::new(),
    ]
    |> @torch.Sequential::new
  let mut accuracy = 0.0
  for datum in mnist_test {
    let input = @torch.tensor(datum.image.map(fn(x) { x.to_double() / 255.0 }))
    let target = Array::make(10, 0.0)
    target[datum.label] = 1.0
    let target = @torch.tensor(target)
    let output = model.forward(input)
    accuracy += (output * target).sum().to_double()
  }
  inspect!(
    accuracy / mnist_test.length().to_double(),
    content="0.10009301315787728",
  )
  let loss = @torch.CrossEntropyLoss::new()
  let optimizer = @torch.SGD::new(model.parameters(), learning_rate=0.0001)
  for datum in mnist_train {
    let input = @torch.tensor(datum.image.map(fn(x) { x.to_double() / 255.0 }))
    let target = Array::make(10, 0.0)
    target[datum.label] = 1.0
    let target = @torch.tensor(target)
    let output = model.forward(input)
    optimizer.zero_grad()
    loss.forward(output, target).backward()
    optimizer.step()
  }
  let mut accuracy = 0.0
  for datum in mnist_test {
    let input = @torch.tensor(datum.image.map(fn(x) { x.to_double() / 255.0 }))
    let target = Array::make(10, 0.0)
    target[datum.label] = 1.0
    let target = @torch.tensor(target)
    let output = model.forward(input)
    accuracy += (output * target).sum().to_double()
  }
  inspect!(
    accuracy / mnist_test.length().to_double(),
    content="0.24793289021264545",
  )
}
