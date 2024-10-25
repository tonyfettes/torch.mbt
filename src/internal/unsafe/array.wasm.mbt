fn UnsafeArray::reinterpret_as[T, U](self : UnsafeArray[T]) -> UnsafeArray[U] = "%identity"

extern "wasm" fn as_pointer(array : FixedArray[Unit]) -> Int =
  #|(func (param i32) (result i32) local.get 0 i32.const 8 i32.add)

pub fn UnsafeArray::reinterpret_as_int[T](self : UnsafeArray[T]) -> Int {
  self.reinterpret_as()._ |> as_pointer()
}
