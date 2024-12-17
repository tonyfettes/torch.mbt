///|
pub fn Tensor::transpose(
  self : Tensor[Float],
  dim0 : Int,
  dim1 : Int
) -> Tensor[Float] {
  Tensor::moveaxis(self, dim0, dim1)
}
