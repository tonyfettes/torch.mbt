pub(all) type UnsafeArray[T] FixedArray[T]

pub fn UnsafeArray::new[T](length : Int) -> UnsafeArray[T] = "%fixedarray.make_uninit"

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

pub fn UnsafeArray::to_json[T : ToJson](self : UnsafeArray[T]) -> Json {
  ToJson::to_json(self._)
}

pub fn UnsafeArray::op_equal[T : Eq](
  self : UnsafeArray[T],
  other : UnsafeArray[T]
) -> Bool {
  if self.length() != other.length() {
    return false
  }
  for i = 0; i < self.length(); i = i + 1 {
    if self[i] != other[i] {
      return false
    }
  }
  true
}

pub fn UnsafeArray::iter[T](self : UnsafeArray[T]) -> Iter[T] {
  let UnsafeArray(array) = self
  array.iter()
}

pub fn UnsafeArray::iter2[T](self : UnsafeArray[T]) -> Iter2[Int, T] {
  let UnsafeArray(array) = self
  array.iter2()
}
