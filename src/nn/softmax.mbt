///|
struct Softmax {} derive(Show, ToJson, FromJson)

///|
pub fn Softmax::new() -> Softmax {
  Softmax::{  }
}

///|
pub fn Softmax::output(_self : Softmax, logger : Logger) -> Unit {
  logger.write_string("Softmax()")
}

///|
pub fn Softmax::forward(
  _self : Softmax,
  input : @tensor.Tensor[Float]
) -> @tensor.Tensor[Float] {
  let exp = input.exp()
  let sum = exp.sum()
  exp / sum
}

///|
pub fn Softmax::parameters(_self : Softmax) -> Iter[@tensor.Tensor[Float]] {
  Iter::empty()
}
