test "array" {
  let a : @array.UnsafeArray[Int] = @array.make(3)
  inspect!(a.length())
}
