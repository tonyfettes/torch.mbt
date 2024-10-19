pub struct AvgPool2d {
  kernel_size : (Int, Int)
  stride : (Int, Int)
} derive(Show, ToJson, FromJson)

pub fn AvgPool2d::new[KernelSize : Shape2d](
  kernel_size : KernelSize,
  ~stride? : Shape2d
) -> AvgPool2d {
  let kernel_size = kernel_size.to_tuple()
  let stride = match stride {
    None => kernel_size
    Some(stride) => stride.to_tuple()
  }
  AvgPool2d::{ kernel_size, stride }
}

pub fn AvgPool2d::forward(self : AvgPool2d, input : Tensor) -> Tensor {
  avg_pool2d(input, kernel_size=self.kernel_size, stride=self.stride)
}

pub fn AvgPool2d::parameters(_self : AvgPool2d) -> Iter[Tensor] {
  Iter::empty()
}

test "AvgPool2d" {
  let input = tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
  let avg_pool2d = AvgPool2d::new(2, stride=(1, 1))
  inspect!(avg_pool2d.forward(input), content="[[[[3, 4], [6, 7]]]]")
}
