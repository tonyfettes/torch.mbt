pub fn Tensor::stack[X : ToTensor + Default](tensors : FixedArray[X]) -> Tensor {
  ToTensor::to_tensor(tensors)
}
