pub struct Data {
  input : FixedArray[Float]
  label : Int
}

let criterion : @torch/nn.CrossEntropyLoss = @torch/nn.CrossEntropyLoss::new()

pub fn train[Model : @torch/nn.Module](
  model : Model,
  optimizer : @torch/optim.SGD,
  batch : FixedArray[Data]
) -> Float {
  let input = batch.map(
    fn(data) {
      @torch.tensor(data.input).reshape(
        [1, 28, 28],
      )
    },
  )
  let input = @torch/tensor.stack(input)
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
  loss.to_float()
}
