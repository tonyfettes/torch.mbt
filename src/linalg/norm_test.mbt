test "norm" {
  let a = @tensor.tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
  @json.inspect!(@linalg.vector_norm(a), content=7.745966911315918)
}
