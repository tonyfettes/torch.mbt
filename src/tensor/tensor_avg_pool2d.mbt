pub fn Tensor::avg_pool2d(
  self : Tensor,
  ~kernel_size : (Int, Int),
  ~stride : (Int, Int)
) -> Tensor {
  if self.shape.length() != 4 {
    abort("Input should be of the shape (N, C, H, W)")
  }
  let kH = kernel_size.0
  let kW = kernel_size.1
  let output_height = (self.shape[2] - kernel_size.0 + stride.0) / stride.0
  let output_width = (self.shape[3] - kernel_size.1 + stride.1) / stride.1
  let output = FixedArray::makei(
    self.shape[0],
    fn(__) {
      FixedArray::makei(
        self.shape[1],
        fn(__) {
          FixedArray::makei(
            output_height,
            fn(__) { FixedArray::make(output_width, tensor(0.0)) },
          )
        },
      )
    },
  )
  for b = 0; b < self.shape[0]; b = b + 1 {
    for c = 0; c < self.shape[1]; c = c + 1 {
      for h = 0; h < output_height; h = h + 1 {
        for w = 0; w < output_width; w = w + 1 {
          let array = FixedArray::make(kH * kW, tensor(0.0))
          for m = 0; m < kH; m = m + 1 {
            for n = 0; n < kW; n = n + 1 {
              array[m * kW + n] = self[b][c][stride.0 * h + m][stride.1 * w + n].to_tensor()
            }
          }
          output[b][c][h][w] = TensorLike::to_tensor(array).sum() /
            tensor((kH * kW).to_double())
        }
      }
    }
  }
  TensorLike::to_tensor(output)
}

test "avg_pool2d" {
  let input = Tensor::new(
    [
      [
        [
          [1.0, 2.0, 3.0, 4.0],
          [2.0, 3.0, 4.0, 5.0],
          [3.0, 4.0, 5.0, 6.0],
          [4.0, 5.0, 6.0, 7.0],
        ],
      ],
    ],
  )
  inspect!(
    avg_pool2d(input, kernel_size=(2, 2), stride=(2, 2)),
    content="[[[[2, 4], [4, 6]]]]",
  )
  inspect!(
    avg_pool2d(input, kernel_size=(2, 2), stride=(1, 1)),
    content="[[[[2, 3, 4], [3, 4, 5], [4, 5, 6]]]]",
  )
}
