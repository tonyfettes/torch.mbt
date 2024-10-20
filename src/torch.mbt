pub typealias Tensor = @tensor.Tensor

pub fn tensor[X : @tensor.TensorValue](x : X, ~requires_grad : Bool = false) -> Tensor {
  x.to_tensor(~requires_grad)
}


pub fn stack[X : @tensor.TensorLike](tensors : FixedArray[X]) -> Tensor {
  @tensor.TensorLike::to_tensor(tensors)
}