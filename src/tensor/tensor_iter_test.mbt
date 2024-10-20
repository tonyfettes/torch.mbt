test "panic Tensor::iter - Scalar" {
  let tensor = @tensor.Tensor::new(1.0)
  inspect!(tensor.iter().collect())
}

test "Tensor::iter - Vector" {
  let tensor = @tensor.Tensor::new([0.0, 1.0, 2.0, 3.0])
  inspect!(tensor.iter().collect(), content="[0, 1, 2, 3]")
}

test "Tensor::iter - Matrix" {
  let tensor = @tensor.Tensor::new([[0.0, 1.0], [2.0, 3.0]])
  inspect!(tensor.iter().collect(), content="[[0, 1], [2, 3]]")
  inspect!(tensor.iter(dim=1).collect(), content="[0, 1, 2, 3]")
  inspect!(tensor.iter(dim=-1).collect(), content="[0, 1, 2, 3]")
}

test "Tensor::iter - Tensor" {
  let tensor = @tensor.Tensor::new(
    [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
  )
  inspect!(
    tensor.iter().collect(),
    content="[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]",
  )
  inspect!(
    tensor.iter(dim=1).collect(),
    content="[[0, 1], [2, 3], [4, 5], [6, 7]]",
  )
  inspect!(tensor.iter(dim=2).collect(), content="[0, 1, 2, 3, 4, 5, 6, 7]")
  inspect!(tensor.iter(dim=-1).collect(), content="[0, 1, 2, 3, 4, 5, 6, 7]")
}

test "panic Tensor::iter - Vector" {
  let tensor = @tensor.Tensor::new([0.0, 1.0, 2.0, 3.0])
  inspect!(tensor.iter(dim=1).collect())
}

test "panic Tensor::iter - Tensor" {
  let tensor = @tensor.Tensor::new(
    [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
  )
  inspect!(tensor.iter(dim=3).collect())
}
