test "TensorView::op_set" {
  let tensor = @tensor.Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  tensor[:][1] = [7.0, 8.0]
  inspect!(tensor, content="[[1, 7, 3], [4, 8, 6]]")
  tensor[:][0] = [0.0, 1.0]
  inspect!(tensor, content="[[0, 7, 3], [1, 8, 6]]")
  tensor[:][2] = [2.0, 3.0]
  inspect!(tensor, content="[[0, 7, 2], [1, 8, 3]]")
  tensor[0][0] = 9.0
  inspect!(tensor, content="[[9, 7, 2], [1, 8, 3]]")
}

test "TensorView::to_tensor" {
  let tensor = @tensor.Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let view = tensor[1]
  let tensor = view.to_tensor()
  inspect!(tensor.value, content="[3, 4, 5]")
  inspect!(tensor.shape, content="[3]")
  inspect!(tensor.block, content="[1]")
  inspect!(tensor.grad, content="[0, 0, 0]")
  inspect!(tensor, content="[3, 4, 5]")
}

test "TensorView::iter" {
  let a = @tensor.tensor([[1.0, 2.0], [3.0, 4.0]])
  @json.inspect!(a[1], content=[3, 4])
  let bs = FixedArray::from_iter(a[1].iter())
  @json.inspect!(bs, content=[3, 4])
}

test "TensorView::iter/shape" {
  let a = @tensor.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
  @json.inspect!(a.shape, content=[2, 2, 2])
  let bs = FixedArray::from_iter(a[1].iter())
  @json.inspect!(bs[0], content=[5, 6])
  @json.inspect!(bs[0].shape, content=[2])
}

test "panic TensorView::iter/scalar" {
  let a = @tensor.tensor([1.0, 2.0, 3.0, 4.0])
  @json.inspect!(a[0], content=1)
  for _ in a[0] {

  }
}
