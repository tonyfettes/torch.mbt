///|
struct Tanh {} derive(Show, ToJson)

///|
pub fn Tanh::new() -> Tanh {
  Tanh::{  }
}

///|
pub fn Tanh::output(_self : Tanh, logger : &Logger) -> Unit {
  logger.write_string("Tanh()")
}

///|
pub fn Tanh::forward(
  _self : Tanh,
  input : @tensor.Tensor[Float]
) -> @tensor.Tensor[Float] {
  input.tanh()
}

///|
pub fn Tanh::parameters(_self : Tanh) -> Iter[@tensor.Tensor[Float]] {
  Iter::empty()
}
