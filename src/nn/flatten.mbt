///|
struct Flatten {
  start : Int
  end : Int?
} derive(Show, ToJson, FromJson)

///|
pub fn Flatten::new(start~ : Int = 1, end? : Int) -> Flatten {
  Flatten::{ start, end }
}

///|
pub fn Flatten::forward(
  self : Flatten,
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input.flatten(start=self.start, end?=self.end)
}

///|
pub fn Flatten::parameters(_self : Flatten) -> Iter[@tensor.Tensor] {
  Iter::empty()
}
