pub trait Shape {
  to_array(Self, ~dim : Int?) -> Array[Int]
}

pub impl Shape for Int with to_array(self : Int, ~dim : Int?) -> Array[Int] {
  match dim {
    None => [self]
    Some(dim) => Array::make(dim, self)
  }
}

pub impl Shape for Array[Int] with to_array(self : Array[Int], ~dim : Int?) -> Array[
  Int,
] {
  match dim {
    Some(dim) =>
      if self.length() != dim {
        abort(
          "Array has \{self.length()} elements, which does not agree with dimension \{dim}",
        )
      }
    None => ()
  }
  self
}

pub trait Shape2d {
  to_array(Self) -> Array[Int]
  to_tuple(Self) -> (Int, Int)
}

pub impl Shape2d for (Int, Int) with to_tuple(self : (Int, Int)) -> (Int, Int) {
  self
}

pub impl Shape2d for (Int, Int) with to_array(self : (Int, Int)) -> Array[Int] {
  [self.0, self.1]
}

pub impl Shape2d for Int with to_tuple(self : Int) -> (Int, Int) {
  (self, self)
}

pub impl Shape2d for Int with to_array(self : Int) -> Array[Int] {
  [self, self]
}
