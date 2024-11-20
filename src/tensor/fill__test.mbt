test "fill_" {
  let tensor = @tensor.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let other = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
  tensor.fill_(other)
  @json.inspect!(tensor, content=[[7, 8, 9], [10, 11, 12]])
}

test "panic fill_/inconsistent_shape" {
  let a = @tensor.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let b = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
  a.fill_(b)
}
