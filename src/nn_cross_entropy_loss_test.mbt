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
