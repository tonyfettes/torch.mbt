///|
pub let float : Float = Float::default()

///|
pub let double : Double = Double::default()

///|
pub fn Tensor::new[Value : ToValue, T : ValueToTensor + Default](
  self : Value,
  dtype~ : T = T::default(),
  requires_grad~ : Bool = false
) -> Tensor[T] {
  dtype |> ignore
  T::value_to_tensor(self.to_value())..requires_grad_(requires_grad)
}

///|
pub fn tensor[Value : ToValue, T : ValueToTensor](
  value : Value,
  requires_grad~ : Bool = false
) -> Tensor[T] {
  T::value_to_tensor(value.to_value())..requires_grad_(requires_grad)
}

test "Tensor::new - Scalar" {
  let tensor : Tensor[Float] = Tensor::new(1.0)
  inspect!(tensor.value, content="[1]")
  inspect!(tensor.shape, content="[]")
  inspect!(tensor.block(), content="[]")
  inspect!(tensor.graph, content="Val")
  inspect!(tensor.refcnt, content="0")
  inspect!(tensor.grad, content="[0]")
  inspect!(tensor, content="1")
}

test "Tensor::new - Vector" {
  let tensor : Tensor[Float] = Tensor::new([0.0, 1.0, 2.0, 3.0])
  inspect!(tensor.value, content="[0, 1, 2, 3]")
  inspect!(tensor.shape, content="[4]")
  inspect!(tensor.block(), content="[1]")
  inspect!(tensor.graph, content="Val")
  inspect!(tensor.refcnt, content="0")
  inspect!(tensor.grad, content="[0, 0, 0, 0]")
  inspect!(tensor, content="[0, 1, 2, 3]")
}

test "Tensor::new - Matrix" {
  let tensor : Tensor[Float] = Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  inspect!(tensor.value, content="[0, 1, 2, 3, 4, 5]")
  inspect!(tensor.shape, content="[2, 3]")
  inspect!(tensor.block(), content="[3, 1]")
  inspect!(tensor.graph, content="Val")
  inspect!(tensor.refcnt, content="0")
  inspect!(tensor.grad, content="[0, 0, 0, 0, 0, 0]")
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}

test "Tensor::new - Tensor" {
  let tensor : Tensor[Float] = Tensor::new([
    [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
    [
      [12.0, 13.0, 14.0, 15.0],
      [16.0, 17.0, 18.0, 19.0],
      [20.0, 21.0, 22.0, 23.0],
    ],
  ])
  inspect!(
    tensor.value,
    content="[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]",
  )
  inspect!(tensor.shape, content="[2, 3, 4]")
  inspect!(tensor.block(), content="[12, 4, 1]")
  inspect!(tensor.graph, content="Val")
  inspect!(tensor.refcnt, content="0")
  inspect!(
    tensor.grad,
    content="[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
  )
  inspect!(
    tensor,
    content="[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]",
  )
}
