pub struct CrossEntropyLoss {}

pub fn CrossEntropyLoss::new() -> CrossEntropyLoss {
  CrossEntropyLoss::{  }
}

pub fn CrossEntropyLoss::output(
  _self : CrossEntropyLoss,
  logger : Logger
) -> Unit {
  logger.write_string("CrossEntropyLoss()")
}

pub fn CrossEntropyLoss::forward(
  _self : CrossEntropyLoss,
  input : Tensor,
  target : Tensor
) -> Tensor {
  let max = input.max()
  let sub_max = input - max
  let log_soft_max = sub_max - sub_max.exp().sum().log()
  let nl = -target * log_soft_max
  nl.sum()
}
