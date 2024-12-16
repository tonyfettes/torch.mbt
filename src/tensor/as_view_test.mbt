test "panic Tensor::op_as_view - Scalar" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(1.0)
  tensor[0:1] |> ignore
}

test "Tensor::op_as_view - Vector" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor([0.0, 1.0, 2.0, 3.0])
  inspect!(tensor[1:3], content="[1, 2]")
  inspect!(tensor[1:], content="[1, 2, 3]")
  inspect!(tensor[:3], content="[0, 1, 2]")
}

test "Tensor::op_as_view - Matrix" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(
    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
  )
  inspect!(tensor[1:2], content="[[3, 4, 5]]")
  inspect!(tensor[1:], content="[[3, 4, 5]]")
  inspect!(tensor[:2], content="[[0, 1, 2], [3, 4, 5]]")
  inspect!(tensor[:], content="[[0, 1, 2], [3, 4, 5]]")
  inspect!(tensor[:][1], content="[1, 4]")
  inspect!(tensor[:][1:], content="[[1, 2], [4, 5]]")
  inspect!(tensor[:][:], content="[[0, 1, 2], [3, 4, 5]]")
}

test "Tensor::op_as_view" {
  let image : @tensor.Tensor[Float] = @tensor.tensor(
    [
      [0.0, 1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0, 7.0],
      [8.0, 9.0, 10.0, 11.0],
      [12.0, 13.0, 14.0, 15.0],
    ],
  )
  inspect!(image[0:3][0:3], content="[[0, 1, 2], [4, 5, 6], [8, 9, 10]]")
  inspect!(
    image[0:3][0:3].to_tensor(),
    content="[[0, 1, 2], [4, 5, 6], [8, 9, 10]]",
  )
}
