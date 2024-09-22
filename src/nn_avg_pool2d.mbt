pub struct AvgPool2d {
  kernel_size : (Int, Int)
  stride : (Int, Int)
} derive(ToJson, FromJson)

pub fn AvgPool2d::new[KernelSize : Shape2d, Stride : Shape2d](kernel_size : KernelSize, ~stride? : Stride) -> AvgPool2d {
  let kernel_size = kernel_size.to_tuple()
  let stride = match stride {
    None => kernel_size
    Some(stride) => stride.to_tuple()
  }
  AvgPool2d::{
    kernel_size,
    stride
  }
}

pub fn AvgPool2d::forward(self : AvgPool2d, input: Tensor) -> Tensor {
  avg_pool2d(input, kernel_size=self.kernel_size, ~stride=self.stride)
}
