test "UnsafeArray::new" {
  let _ : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray::new(3)

}

test "UnsafeArray::make" {
  let a : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray::make(3, 0)
  inspect!(a.length(), content="3")
}

test "UnsafeArray::from_fixedarray" {
  let a : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray([1, 2, 3])
  inspect!(a[0], content="1")
  inspect!(a[1], content="2")
  inspect!(a[2], content="3")
}

test "UnsafeArray::op_equal" {
  let a : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray([1, 2, 3])
  let b : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray([1, 2, 3])
  let c : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray([1, 2, 4])
  let d : @unsafe.UnsafeArray[Int] = @unsafe.UnsafeArray([1, 2])
  inspect!(a == b, content="true")
  inspect!(a == c, content="false")
  inspect!(a == d, content="false")
}
