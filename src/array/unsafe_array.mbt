pub type UnsafeArray[_]

pub fn UnsafeArray::op_set[T](
  self : UnsafeArray[T],
  index : UInt,
  value : T
) -> Unit = "%fixedarray.unsafe_set"

pub fn UnsafeArray::op_get[T](self : UnsafeArray[T], index : UInt) -> T = "%fixedarray.unsafe_get"

pub fn UnsafeArray::make[T](length : UInt) -> UnsafeArray[T] = "%fixedarray.make_uninit"

pub fn UnsafeArray::length[T](self : UnsafeArray[T]) -> UInt = "%fixedarray.length"
