test "norm" {
  let a = @tensor.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
  @json.inspect!(@linalg.vector_norm(a), content=7.745966911315918)
}
