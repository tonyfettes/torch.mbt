///|
pub fn Tensor::max_pool2d(
  self : Tensor[Float],
  kernel_size~ : (Int, Int),
  stride~ : (Int, Int)
) -> Tensor[Float] {
  let self_shape = self.shape
  let self_block = self.block()
  if self_shape.length() != 4 {
    abort("Input should be of the shape (N, C, H, W)")
  }
  let iN = self_shape[0]
  let iC = self_shape[1]
  let iH = self_shape[2]
  let iW = self_shape[3]
  let kH = kernel_size.0
  let kW = kernel_size.1
  let kA = (kH * kW).to_float()
  let oH = (iH - kH + stride.0) / stride.0
  let oW = (iW - kW + stride.1) / stride.1
  let shape : FixedArray[Int] = [iN, iC, oH, oW]
  let block = compute_block_size(shape)
  let total = shape[0] * block[0]
  let value = @unsafe.UnsafeArray::make(total, (0.0 : Float))
  let index = @unsafe.UnsafeArray::make(total, 0)
  fn index4d(
    block : @unsafe.UnsafeArray[Int],
    n : Int,
    c : Int,
    h : Int,
    w : Int
  ) -> Int {
    return n * block[0] + c * block[1] + h * block[2] + w
  }

  for n = 0; n < iN; n = n + 1 {
    for c = 0; c < iC; c = c + 1 {
      for h = 0; h < oH; h = h + 1 {
        for w = 0; w < oW; w = w + 1 {
          let mut maxima : Float = @float.min_value
          let mut maxima_index : Int = -1
          for i = 0; i < kH; i = i + 1 {
            for j = 0; j < kW; j = j + 1 {
              let index = index4d(
                self_block,
                n,
                c,
                stride.0 * h + i,
                stride.1 * w + j,
              )
              maxima = @math.maximum(maxima, self.value[index])
              maxima_index = index
            }
          }
          let value_index = index4d(block, n, c, h, w)
          value[value_index] = maxima / kA
          index[value_index] = maxima_index
        }
      }
    }
  }
  Tensor::{
    value,
    shape,
    graph: MaxPool2d(self, index),
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
