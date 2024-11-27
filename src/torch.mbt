pub(all) typealias Tensor = @tensor.Tensor

pub fn tensor[X : @tensor.ToTensor](
  x : X,
  requires_grad~ : Bool = false
) -> Tensor {
  tensor(x, requires_grad~)
}

/// TODO: This should be moved into
/// @alert deprecated "Use `distributions` package to initialize an random tensor."
pub fn randn[Shape : @shape.Shape](
  shape : Shape,
  requires_grad~ : Bool = false
) -> Tensor {
  let sample = @distributions.standard_normal.sample(shape)
  if requires_grad {
    sample.requires_grad_(true)
  }
  sample
}
