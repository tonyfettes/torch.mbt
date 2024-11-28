test "constant_" {
  let tensor = @tensor.empty([2, 3])
  @init.constant_(tensor, 3.0)
  @json.inspect!(tensor.value, content=[3, 3, 3, 3, 3, 3])
}

test "ones_" {
  let tensor = @tensor.empty([2, 3])
  @init.ones_(tensor)
  @json.inspect!(tensor.value, content=[1, 1, 1, 1, 1, 1])
}

test "zeros_" {
  let tensor = @tensor.empty([2, 3])
  @init.zeros_(tensor)
  @json.inspect!(tensor.value, content=[0, 0, 0, 0, 0, 0])
}
