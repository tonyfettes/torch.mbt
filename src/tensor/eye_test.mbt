test "eye" {
  @json.inspect!(@tensor.eye(3), content=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
}

test "eye/rectangle" {
  @json.inspect!(@tensor.eye(2, m=3), content=[[1, 0, 0], [0, 1, 0]])
  @json.inspect!(@tensor.eye(3, m=2), content=[[1, 0], [0, 1], [0, 0]])
}
