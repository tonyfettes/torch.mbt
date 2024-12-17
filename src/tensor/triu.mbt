///|
pub fn triu[Dtype : Default + @unsafe.Primitive](
  self : Tensor[Dtype],
  diagonal~ : Int = 0
) -> Tensor[Dtype] {
  if @debug.DEBUG {
    if self.shape.length() != 2 {
      abort("Triu only supports 2D tensors")
    }
  }
  let shape = self.shape
  let block = self.block
  let value = @unsafe.UnsafeArray::new(self.value.length())
  let index = @unsafe.UnsafeArray::new(self.value.length())
  for i in 0..<shape[0] {
    for j in 0..<shape[1] {
      let pos = i * block[0] + j
      if j - i >= diagonal {
        value[pos] = self.value[pos]
        index[pos] = pos
      } else {
        value[pos] = Dtype::default()
      }
    }
  }
  Tensor::{
    value,
    shape,
    block,
    refcnt: 0,
    graph: Get(self, index),
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}
