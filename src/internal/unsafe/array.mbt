pub type UnsafeArray[T] FixedArray[T]

pub fn UnsafeArray::op_set[T](
  self : UnsafeArray[T],
  index : Int,
  value : T
) -> Unit = "%fixedarray.unsafe_set"

pub fn UnsafeArray::op_get[T](self : UnsafeArray[T], index : Int) -> T = "%fixedarray.unsafe_get"

pub fn UnsafeArray::make[T](length : Int, init : T) -> UnsafeArray[T] = "%fixedarray.make"

pub fn UnsafeArray::length[T](self : UnsafeArray[T]) -> Int = "%fixedarray.length"

pub impl[T : Show] Show for UnsafeArray[T] with output(
  self : UnsafeArray[T],
  logger : Logger
) {
  Show::output(self._, logger)
}

pub fn UnsafeArray::to_string[T : Show](self : UnsafeArray[T]) -> String {
  self._.to_string()
}
