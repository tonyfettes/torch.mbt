///|
struct ReLU {} derive(Show, ToJson, FromJson)

///|
pub fn ReLU::new() -> ReLU {
  ReLU::{  }
}

///|
pub fn ReLU::output(_self : ReLU, logger : Logger) -> Unit {
  logger.write_string("ReLU()")
}

///|
pub fn ReLU::forward(_self : ReLU, input : @tensor.Tensor[Float]) -> @tensor.Tensor[Float] {
  @functional.relu(input)
}

///|
pub fn ReLU::parameters(_self : ReLU) -> Iter[@tensor.Tensor[Float]] {
  Iter::empty()
}
