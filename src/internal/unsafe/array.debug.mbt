///|
pub fn UnsafeArray::op_set[T](
  self : UnsafeArray[T],
  index : Int,
  value : T
) -> Unit = "%fixedarray.set"

///|
pub fn UnsafeArray::op_get[T](self : UnsafeArray[T], index : Int) -> T = "%fixedarray.get"
