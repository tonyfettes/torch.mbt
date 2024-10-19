pub fn correlate2d(
  image : Tensor,
  kernel : Tensor,
  ~stride : Int = 1,
  ~padding : Shape2d = 0
) -> Tensor {
  let image_height = image.shape[0]
  let image_width = image.shape[1]
  let kernel_height = kernel.shape[0]
  let kernel_width = kernel.shape[1]
  let (padding_height, padding_width) = padding.to_tuple()
  let output_height = (
      image_height + 2 * padding_height - kernel_height + stride
    ) /
    stride
  let output_width = (image_width + 2 * padding_width - kernel_width + stride) /
    stride
  FixedArray::makei(
    output_height,
    fn(i) {
      FixedArray::makei(
        output_width,
        fn(j) {
          let i_start = i * stride - padding_height
          let i_end = i_start + kernel_height
          let (pi_start, ki_start) = if i_start < 0 {
            (0, -i_start)
          } else {
            (i_start, 0)
          }
          let (pi_end, ki_end) = if i_end > image_height {
            (image_height, image_height - pi_start)
          } else {
            (i_end, kernel_height)
          }
          let j_start = j * stride - padding_width
          let j_end = j_start + kernel_width
          let (pj_start, kj_start) = if j_start < 0 {
            (0, -j_start)
          } else {
            (j_start, 0)
          }
          let (pj_end, kj_end) = if j_end > image_width {
            (image_width, image_width - pj_start)
          } else {
            (j_end, kernel_width)
          }
          let patch = image[pi_start:pi_end][pj_start:pj_end].to_tensor()
          let kernel = kernel[ki_start:ki_end][kj_start:kj_end].to_tensor()
          (patch * kernel).sum()
        },
      )
      |> Tensor::stack()
    },
  )
  |> Tensor::stack()
}
