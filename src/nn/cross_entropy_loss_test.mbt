test "CrossEntropyLoss" {
  let input : @tensor.Tensor[Float] = @tensor.tensor([0.0, 1.0e3, 0.0])
  let target : @tensor.Tensor[Float] = @tensor.tensor([0.3, 0.3, 0.4])
  let loss = @nn.CrossEntropyLoss::new()
  let output = loss.forward(input, target)
  inspect!(output, content="700")
}

test "CrossEntropyLoss/Random" {
  let input : @tensor.Tensor[Float] = @tensor.tensor([
    -0.14630347333285626, -0.11041421512364283, -0.12559252186003866, -0.13513121430614242,
    -0.1788202792999719, -0.1412897765465771, -0.12746655654191671, -0.12665480060556913,
    -0.14233145825454024, -0.1360269781704941,
  ])
  let target = {
    let target = FixedArray::make(10, 0.0)
    target[6] = 1.0
    @tensor.tensor(target)
  }
  let criterion = @nn.CrossEntropyLoss::new()
  let output = criterion.forward(input, target)
  inspect!(output, content="2.293193817138672")
}

test "CrossEntropyLoss/Backward" {
  let input : @tensor.Tensor[Float] = @tensor.tensor([
    -0.14630347333285626, -0.11041421512364283, -0.12559252186003866, -0.13513121430614242,
    -0.1788202792999719, -0.1412897765465771, -0.12746655654191671, -0.12665480060556913,
    -0.14233145825454024, -0.1360269781704941,
  ])
  let target = {
    let target = FixedArray::make(10, 0.0)
    target[6] = 1.0
    @tensor.tensor(target)
  }
  let criterion = @nn.CrossEntropyLoss::new()
  let optimizer = @optim.SGD::new(Iter::singleton(input), learning_rate=0.1)
  let output = criterion.forward(input, target)
  optimizer.zero_grad()
  output.backward()
  optimizer.step()
  @json.inspect!(input, content=[
    -0.15620946884155273, -0.12068218737840652, -0.13570579886436462, -0.14514848589897156,
    -0.18840934336185455, -0.15124554932117462, -0.03756091743707657, -0.13675735890865326,
    -0.15227685868740082, -0.14603528380393982,
  ])
}

test "CrossEntropyLoss/Batch" {
  let input : @tensor.Tensor[Float] = @tensor.tensor(
    FixedArray::makei(2 * 4 * 2 * 2, fn(i) { i.to_double() }),
  ).reshape([2, 4, 2, 2])
  let target = @tensor.zeros([2, 4, 2, 2])
  target[0][:][0][0] = [1.0, 0.0, 0.0, 0.0]
  target[0][:][0][1] = [0.0, 1.0, 0.0, 0.0]
  target[0][:][1][0] = [0.0, 0.0, 1.0, 0.0]
  target[0][:][1][1] = [0.0, 0.0, 0.0, 1.0]
  target[1][:][0][0] = [0.5, 0.0, 0.5, 0.0]
  target[1][:][0][1] = [0.0, 0.5, 0.0, 0.5]
  target[1][:][1][0] = [0.5, 0.0, 0.0, 0.5]
  target[1][:][1][1] = [0.0, 0.5, 0.5, 0.0]
  @json.inspect!(target, content=[
    [[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]],
    [
      [[0.5, 0], [0.5, 0]],
      [[0, 0.5], [0, 0.5]],
      [[0.5, 0], [0, 0.5]],
      [[0, 0.5], [0.5, 0]],
    ],
  ])
  let criterion = @nn.CrossEntropyLoss::new()
  let output = criterion.forward(input, target)
  @json.inspect!(output, content=6.0184855461120605)
}
