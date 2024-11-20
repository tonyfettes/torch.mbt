pub(all) struct MSELoss {}

pub fn MSELoss::new() -> MSELoss {
  MSELoss::{  }
}

pub fn MSELoss::output(_self : MSELoss, logger : Logger) -> Unit {
  logger.write_string("MSELoss()")
}

pub fn MSELoss::forward(
  _self : MSELoss,
  input : @tensor.Tensor,
  target : @tensor.Tensor
) -> @tensor.Tensor {
  if input.length() != target.length() {
    abort("Input and target tensors should have the same length")
  }
  let diff = input - target
  diff.pow(2).mean()
}
