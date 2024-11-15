pub fn Tensor::avg_pool2d(
  self : Tensor,
  kernel_size~ : (Int, Int),
  stride~ : (Int, Int)
) -> Tensor {
  if self.shape.length() != 4 {
    abort("Input should be of the shape (N, C, H, W)")
  }
  let iN = self.shape[0]
  let iC = self.shape[1]
  let iH = self.shape[2]
  let iW = self.shape[3]
  let kH = kernel_size.0
  let kW = kernel_size.1
  let kA = (kH * kW).to_float()
  let oH = (iH - kH + stride.0) / stride.0
  let oW = (iW - kW + stride.1) / stride.1
  let shape : FixedArray[Int] = [iN, iC, oH, oW]
  let block = compute_block_size(shape)
  let total = shape[0] * block[0]
  let value = @unsafe.UnsafeArray::make(total, (0.0 : Float))
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
          let mut kernel : Float = 0.0
          for i = 0; i < kH; i = i + 1 {
            for j = 0; j < kW; j = j + 1 {
              let index = index4d(
                self.block,
                n,
                c,
                stride.0 * h + i,
                stride.1 * w + j,
              )
              kernel += self.value[index]
            }
          }
          value[index4d(block, n, c, h, w)] = kernel / kA
        }
      }
    }
  }
  Tensor::{
    value,
    shape,
    block,
    graph: AvgPool2d(self, kernel_size~, stride~),
    ref: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
