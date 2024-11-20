pub(all) struct ReLU {} derive(Show, ToJson, FromJson)

pub fn ReLU::new() -> ReLU {
  ReLU::{  }
}

pub fn ReLU::output(_self : ReLU, logger : Logger) -> Unit {
  logger.write_string("ReLU()")
}

pub fn ReLU::forward(_self : ReLU, input : @tensor.Tensor) -> @tensor.Tensor {
  input.reLU()
}

pub fn ReLU::parameters(_self : ReLU) -> Iter[@tensor.Tensor] {
  Iter::empty()
}

pub(all) struct ReLU6 {}
