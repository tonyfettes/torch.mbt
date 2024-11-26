test "gather" {
  let t = @tensor.tensor([[1.0, 2.0], [3.0, 4.0]])
  @json.inspect!(
    @tensor.gather(t, 1, @tensor.tensor([[0.0, 0.0], [1.0, 0.0]])),
    content=[[1, 1], [4, 3]],
  )
}
