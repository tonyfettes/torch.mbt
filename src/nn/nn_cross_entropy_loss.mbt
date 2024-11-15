pub(all) struct CrossEntropyLoss {
  reduction : Reduction
}

pub(all) enum Reduction {
  None
  Mean
  Sum
}

pub fn CrossEntropyLoss::new(reduction~ : Reduction = Mean) -> CrossEntropyLoss {
  CrossEntropyLoss::{ reduction, }
}

pub fn CrossEntropyLoss::output(
  _self : CrossEntropyLoss,
  logger : Logger
) -> Unit {
  logger.write_string("CrossEntropyLoss()")
}

pub fn CrossEntropyLoss::forward(
  self : CrossEntropyLoss,
  input : @tensor.Tensor,
  target : @tensor.Tensor
) -> @tensor.Tensor {
  let input_shape = input.shape
  if input_shape != target.shape {
    abort(
      "Input shape \{input.shape} is not consistent with target shape \{target.shape}",
    )
  }
  let dimension = input_shape.length()
  fn compute1d(
    input : @tensor.Tensor,
    target : @tensor.Tensor
  ) -> @tensor.Tensor {
    let max = input.max()
    let sub_max = input - max
    let log_soft_max = sub_max - sub_max.exp().sum().log()
    let nl = -target * log_soft_max
    nl.sum()
  }

  let output = match dimension {
    1 => compute1d(input, target)
    2 =>
      FixedArray::makei(
        input_shape[0],
        fn(i) { compute1d(input[i][:].to_tensor(), target[i][:].to_tensor()) },
      )
      |> @tensor.stack()
    dimension =>
      if dimension <= 0 {
        abort(
          "Cannot compute cross-entropy loss for tensor with a dimension less than 1",
        )
      } else {
        let input = input.flatten(start=2)
        let target = target.flatten(start=2)
        let output = FixedArray::makei(
            input.shape[0],
            fn(i) {
              FixedArray::makei(
                input.shape[2],
                fn(j) {
                  compute1d(
                    input[i][:][j].to_tensor(),
                    target[i][:][j].to_tensor(),
                  )
                },
              )
              |> @tensor.stack()
            },
          )
          |> @tensor.stack()
        let output_shape = FixedArray::make(dimension - 1, input_shape[0])
        for i in 1..<(dimension - 1) {
          output_shape[i] = input_shape[i + 1]
        }
        output.reshape(output_shape)
      }
  }
  match self.reduction {
    None => output
    Sum => output.sum()
    Mean => output.mean()
  }
}
