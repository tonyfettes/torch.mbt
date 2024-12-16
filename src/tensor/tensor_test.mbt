test "Tensor::new - Scalar" {
  let value = 1.0
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(value)
  inspect!(tensor, content="1")
}

test "Tensor::new - Vector" {
  let value = [0.0, 1.0, 2.0, 3.0]
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(value)
  inspect!(tensor, content="[0, 1, 2, 3]")
}

test "Tensor::new - Matrix" {
  let value = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(value)
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}

test "Tensor::new - Tensor" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(
    [
      [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
      [
        [12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0],
        [20.0, 21.0, 22.0, 23.0],
      ],
    ],
  )
  inspect!(
    tensor,
    content="[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]",
  )
}

test "panic Tensor::op_get - Scalar" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(1.0)
  tensor[0] |> ignore
}

test "Tensor::op_get - Vector" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor([0.0, 1.0, 2.0, 3.0])
  let slice = tensor[1]
  inspect!(slice, content="1")
}

test "Tensor::op_get - Matrix" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(
    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
  )
  let view = tensor[1]
  inspect!(view, content="[3, 4, 5]")
}

test "TensorView::to_tensor" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(
    [
      [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
    ],
  )
  inspect!(
    tensor[:][0].to_tensor(),
    content="[[0, 1, 2], [1, 2, 3], [2, 3, 4]]",
  )
  inspect!(tensor[0][:].to_tensor(), content="[[0, 1, 2], [3, 4, 5]]")
  inspect!(tensor[0][:][1].to_tensor(), content="[1, 4]")
  inspect!(tensor[:][1][1].to_tensor(), content="[4, 5, 6]")
  inspect!(tensor[:1][1][:].to_tensor(), content="[[3, 4, 5]]")
}

test "Tensor::permute" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(
    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
  )
  inspect!(tensor.permute([1, 0]), content="[[0, 3], [1, 4], [2, 5]]")
}

test "TensorView::to_json" {
  let tensor : @tensor.Tensor[Float] = @tensor.tensor(
    [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]],
  )
  let view = tensor[1][:][2]
  @json.inspect!(view, content=[8, 11])
}
