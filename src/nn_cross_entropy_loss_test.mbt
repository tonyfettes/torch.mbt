test "CrossEntropyLoss" {
  let input = @torch.tensor([0.0, 1.0e3, 0.0])
  let target = @torch.tensor([0.3, 0.3, 0.4])
  let loss = @torch.CrossEntropyLoss::new()
  let output = loss.forward(input, target)
  inspect!(output, content="700")
}

test "CrossEntropyLoss/Random" {
  let input = @torch.tensor(
    [
      -0.14630347333285626, -0.11041421512364283, -0.12559252186003866, -0.13513121430614242,
      -0.1788202792999719, -0.1412897765465771, -0.12746655654191671, -0.12665480060556913,
      -0.14233145825454024, -0.1360269781704941,
    ],
  )
  let target = {
    let target = FixedArray::make(10, 0.0)
    target[6] = 1.0
    @torch.tensor(target)
  }
  let criterion = @torch.CrossEntropyLoss::new()
  let output = criterion.forward(input, target)
  inspect!(output, content="2.293193741636991")
}

test "CrossEntropyLoss/Backward" {
  let input = @torch.tensor(
    [
      -0.14630347333285626, -0.11041421512364283, -0.12559252186003866, -0.13513121430614242,
      -0.1788202792999719, -0.1412897765465771, -0.12746655654191671, -0.12665480060556913,
      -0.14233145825454024, -0.1360269781704941,
    ],
  )
  let target = {
    let target = FixedArray::make(10, 0.0)
    target[6] = 1.0
    @torch.tensor(target)
  }
  let criterion = @torch.CrossEntropyLoss::new()
  let optimizer = @torch.SGD::new(Iter::singleton(input), learning_rate=0.1)
  let output = criterion.forward(input, target)
  optimizer.zero_grad()
  output.backward()
  optimizer.step()
  @json.inspect!(
    input,
    content=[
      -0.15620946237082442, -0.12068217941940952, -0.13570581265456363, -0.14514849615800837,
      -0.18840933792174988, -0.15124555592247957, -0.03756091242662528, -0.13675735397031713,
      -0.15227687227681524, -0.14603529092095632,
    ],
  )
}

test "CrossEntropyLoss/Batch" {
  let input = @torch.tensor(
    FixedArray::makei(2 * 4 * 2 * 2, fn(i) { i.to_double() }),
  ).reshape([2, 4, 2, 2])
  let target = @torch.zeros([2, 4, 2, 2])
  target[0][:][0][0] = [1.0, 0.0, 0.0, 0.0]
  target[0][:][0][1] = [0.0, 1.0, 0.0, 0.0]
  target[0][:][1][0] = [0.0, 0.0, 1.0, 0.0]
  target[0][:][1][1] = [0.0, 0.0, 0.0, 1.0]
  target[1][:][0][0] = [0.5, 0.0, 0.5, 0.0]
  target[1][:][0][1] = [0.0, 0.5, 0.0, 0.5]
  target[1][:][1][0] = [0.5, 0.0, 0.0, 0.5]
  target[1][:][1][1] = [0.0, 0.5, 0.5, 0.0]
  @json.inspect!(
    target,
    content=[
      [[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]],
      [
        [[0.5, 0], [0.5, 0]],
        [[0, 0.5], [0, 0.5]],
        [[0.5, 0], [0, 0.5]],
        [[0, 0.5], [0.5, 0]],
      ],
    ],
  )
  let criterion = @torch.CrossEntropyLoss::new()
  let output = criterion.forward(input, target)
  @json.inspect!(
    output,
    content=6.018485334290704,
  )
}
