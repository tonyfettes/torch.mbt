///|
typealias Tensor[Dtype] = @tensor.Tensor[Dtype]

///|
pub let float : Float = 0.0

///|
pub fn tensor[Value : @tensor.ToValue, Dtype : @tensor.ValueToTensor + Default](
  value : Value,
  dtype~ : Dtype = Dtype::default(),
  requires_grad~ : Bool = false
) -> Tensor[Dtype] {
  dtype |> ignore
  @tensor.tensor(value, requires_grad~)
}

///| TODO: This should be moved into
/// @alert deprecated "Use `distributions` package to initialize an random tensor."
pub fn randn[Shape : @shape.Shape](
  shape : Shape,
  requires_grad~ : Bool = false
) -> @tensor.Tensor[Float] {
  let sample = @distributions.standard_normal.sample(shape)
  if requires_grad {
    sample.requires_grad_(true)
  }
  sample
}
