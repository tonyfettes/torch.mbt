///|
pub fn UnsafeArray::op_set[T](
  self : UnsafeArray[T],
  index : Int,
  value : T
) -> Unit = "%fixedarray.unsafe_set"

///|
pub fn UnsafeArray::op_get[T](self : UnsafeArray[T], index : Int) -> T = "%fixedarray.unsafe_get"
