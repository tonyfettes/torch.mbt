///|
pub fn correlate2d(
  image : Tensor[Float],
  kernel : Tensor[Float],
  stride~ : &@shape.Shape2d = 1,
  padding~ : &@shape.Shape2d = 0
) -> Tensor[Float] {
  let image_height = image.shape[0]
  let image_width = image.shape[1]
  let image_value = image.value
  let kernel_height = kernel.shape[0]
  let kernel_width = kernel.shape[1]
  let kernel_value = kernel.value
  let (stride_height, stride_width) = stride.to_tuple()
  let (padding_height, padding_width) = padding.to_tuple()
  let output_height = (
      image_height + 2 * padding_height - kernel_height + stride_height
    ) /
    stride_height
  let output_width = (
      image_width + 2 * padding_width - kernel_width + stride_width
    ) /
    stride_width
  let shape : FixedArray[_] = [output_height, output_width]
  let block = compute_block_size(shape)
  let total = output_height * output_width
  let value = @unsafe.UnsafeArray::make(total, (0.0 : Float))
  for i in 0..<output_height {
    for j in 0..<output_width {
      let i_start = i * stride_height - padding_height
      let pi_start = @math.maximum(0, i_start)
      let ki_start = @math.maximum(0, -i_start)
      let i_length = @math.minimum(
        kernel_height - ki_start,
        image_height - pi_start,
      )
      let j_start = j * stride_width - padding_width
      let pj_start = @math.maximum(0, j_start)
      let kj_start = @math.maximum(0, -j_start)
      let j_length = @math.minimum(
        kernel_width - kj_start,
        image_width - pj_start,
      )
      for k in 0..<i_length {
        let pj_base = (pi_start + k) * image_width
        let kj_base = (ki_start + k) * kernel_width
        for l in 0..<j_length {
          let image_value = image_value[pj_base + pj_start + l]
          let kernel_value = kernel_value[kj_base + kj_start + l]
          value[i * output_width + j] += image_value * kernel_value
        }
      }
    }
  }
  Tensor::{
    value,
    shape,
    block,
    graph: Correlate2d(
      image~,
      kernel~,
      stride=(stride_height, stride_width),
      padding=(padding_height, padding_width),
    ),
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
