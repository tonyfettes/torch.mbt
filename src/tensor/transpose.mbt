///|
pub fn Tensor::transpose(self : Tensor, dim0 : Int, dim1 : Int) -> Tensor {
  Tensor::moveaxis(self, [dim0, dim1], [dim1, dim0])
}
