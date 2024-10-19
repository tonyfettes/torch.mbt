pub fn correlate2d(
  image : Tensor,
  kernel : Tensor,
  ~stride : Int = 1,
  ~padding : Int = 0
) -> Tensor {
  let image_height = image.shape[0]
  let image_width = image.shape[1]
  let kernel_height = kernel.shape[0]
  let kernel_width = kernel.shape[1]
  let output_height = (image_height + 2 * padding - kernel_height + stride) /
    stride
  let output_width = (image_width + 2 * padding - kernel_width + stride) /
    stride
  FixedArray::makei(
    output_height,
    fn(i) {
      FixedArray::makei(
        output_width,
        fn(j) {
          let i_start = @math.maximum(0, i * stride - padding)
          let i_end = @math.minimum(
            i * stride + kernel_height - padding,
            image_height,
          )
          let j_start = @math.maximum(0, j * stride - padding)
          let j_end = @math.minimum(
            j * stride + kernel_width - padding,
            image_width,
          )
          let patch = image[i_start:i_end][j_start:j_end].to_tensor()
          (patch * kernel).sum()
        },
      )
      |> Tensor::stack()
    },
  )
  |> Tensor::stack()
}
