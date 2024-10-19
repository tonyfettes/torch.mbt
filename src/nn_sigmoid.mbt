struct Sigmoid {} derive(Show, ToJson, FromJson)

pub fn Sigmoid::new() -> Sigmoid {
  Sigmoid::{  }
}

pub fn Sigmoid::forward(_self : Sigmoid, input : Tensor) -> Tensor {
  input.sigmoid()
}

pub fn Sigmoid::parameters(_self : Sigmoid) -> Iter[Tensor] {
  return Iter::empty()
}
