pub struct AvgPool2d {
  kernel_size : (Int, Int)
  stride : (Int, Int)
} derive(Show, ToJson, FromJson)

pub fn AvgPool2d::new[KernelSize : @shape.Shape2d](
  kernel_size : KernelSize,
  ~stride? : @shape.Shape2d
) -> AvgPool2d {
  let kernel_size = kernel_size.to_tuple()
  let stride = match stride {
    None => kernel_size
    Some(stride) => stride.to_tuple()
  }
  AvgPool2d::{ kernel_size, stride }
}

pub fn AvgPool2d::forward(self : AvgPool2d, input : @tensor.Tensor) -> @tensor.Tensor {
  @tensor.avg_pool2d(input, kernel_size=self.kernel_size, stride=self.stride)
}

pub fn AvgPool2d::parameters(_self : AvgPool2d) -> Iter[@tensor.Tensor] {
  Iter::empty()
}

pub impl ToPyTorchSource for AvgPool2d with transpile(
  self : AvgPool2d,
  transpiler : PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model(
    "\{variable} = torch.nn.AvgPool2d(kernel_size=\{self.kernel_size}, stride=\{self.stride})",
  )
  variable
}
