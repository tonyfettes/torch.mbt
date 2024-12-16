///|
struct Sigmoid {} derive(Show, ToJson, FromJson)

///|
pub fn Sigmoid::new() -> Sigmoid {
  Sigmoid::{  }
}

///|
pub fn Sigmoid::forward(
  _self : Sigmoid,
  input : @tensor.Tensor[Float]
) -> @tensor.Tensor[Float] {
  input.sigmoid()
}

///|
pub fn Sigmoid::parameters(_self : Sigmoid) -> Iter[@tensor.Tensor[Float]] {
  return Iter::empty()
}
