pub(open) trait Shape {
  to_fixed_array(Self) -> FixedArray[Int]
  iter(Self) -> Iter[Int]
}

pub impl Shape for Unit with to_fixed_array(_self : Unit) -> FixedArray[Int] {
  FixedArray::make(0, 0)
}

pub impl Shape for Unit with iter(_self : Unit) -> Iter[Int] { Iter::empty() }

pub impl Shape for Int with to_fixed_array(self : Int) -> FixedArray[Int] {
  [self]
}

pub impl Shape for Int with iter(self : Int) -> Iter[Int] {
  Iter::singleton(self)
}

pub impl Shape for Array[Int] with to_fixed_array(self : Array[Int]) -> FixedArray[
  Int,
] {
  FixedArray::from_array(self)
}

pub impl Shape for FixedArray[Int] with to_fixed_array(self : FixedArray[Int]) -> FixedArray[
  Int,
] {
  self
}

pub(readonly) trait Shape2d {
  to_fixed_array(Self) -> FixedArray[Int]
  to_tuple(Self) -> (Int, Int)
}

impl Shape2d with to_fixed_array(self : Self) -> FixedArray[Int] {
  let (s0, s1) = self.to_tuple()
  [s0, s1]
}

pub impl Shape2d for (Int, Int) with to_tuple(self : (Int, Int)) -> (Int, Int) {
  self
}

pub impl Shape2d for Int with to_tuple(self : Int) -> (Int, Int) {
  (self, self)
}
