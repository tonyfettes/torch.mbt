pub(all) typealias Tensor = @tensor.Tensor

pub fn tensor[X : @tensor.ToTensor](
  x : X,
  requires_grad~ : Bool = false
) -> Tensor {
  tensor(x, requires_grad~)
}
