///|
pub fn Tensor::stack[X : ToTensor + Default](tensors : FixedArray[X]) -> Tensor[Float] {
  ToTensor::to_tensor(tensors)
}
