pub fn Tensor::stack[X : TensorLike](tensors : FixedArray[X]) -> Tensor {
  TensorLike::to_tensor(tensors)
}
