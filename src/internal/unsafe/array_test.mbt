test "UnsafeArray/make" {
  let a : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray::make(3, 0)
  inspect!(a.length(), content="3")
}

test "UnsafeArray/from_fixedarray" {
  let a : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray([1, 2, 3])
  inspect!(a[0], content="1")
  inspect!(a[1], content="2")
  inspect!(a[2], content="3")
}
