pub struct Data {
  input : FixedArray[Int]
  label : Int
}

let criterion : @torch.CrossEntropyLoss = @torch.CrossEntropyLoss::new()

pub fn train[Model : @torch.Module](
  model : Model,
  optimizer : @torch.SGD,
  batch : FixedArray[Data]
) -> Double {
  let input = batch.map(
    fn(data) {
      @torch.tensor(data.input.map(fn(x) { x.to_double() / 255.0 })).reshape(
        [1, 28, 28],
      )
    },
  )
  let input = @torch.stack(input)
  let target = batch.map(
      fn(data) {
        let target = FixedArray::make(10, 0.0)
        target[data.label] = 1.0
        target
      },
    )
    |> @torch.tensor()
  let output = model.forward(input)
  optimizer.zero_grad()
  let loss = criterion.forward(output, target)
  loss.backward()
  optimizer.step()
  loss.to_double()
}
