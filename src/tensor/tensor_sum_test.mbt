test "Tensor::sum - scalar" {
  let tensor = @tensor.tensor(1.0)
  let sum = tensor.sum()
  inspect!(sum.value, content="[1]")
  inspect!(sum.shape, content="[]")
  inspect!(sum.block, content="[]")
  inspect!(sum.graph, content="Sum(1, [[0]])")
  inspect!(sum.ref, content="0")
  inspect!(sum.grad, content="[0]")
  inspect!(sum, content="1")
}

test "Tensor::sum - Vector" {
  let tensor = @tensor.tensor([0.0, 1.0, 2.0, 3.0])
  let sum = tensor.sum()
  inspect!(sum.value, content="[6]")
  inspect!(sum.shape, content="[]")
  inspect!(sum.block, content="[]")
  inspect!(sum.graph, content="Sum([0, 1, 2, 3], [[0, 1, 2, 3]])")
  inspect!(sum.ref, content="0")
  inspect!(sum.grad, content="[0]")
  inspect!(sum, content="6")
}

test "Tensor::sum - Matrix" {
  let tensor = @tensor.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let sum = tensor.sum(dim=0)
  inspect!(sum.value, content="[3, 5, 7]")
  inspect!(sum.shape, content="[3]")
  inspect!(sum.block, content="[1]")
  inspect!(
    sum.graph,
    content="Sum([[0, 1, 2], [3, 4, 5]], [[0, 3], [1, 4], [2, 5]])",
  )
  inspect!(sum.ref, content="0")
  inspect!(sum.grad, content="[0, 0, 0]")
  inspect!(sum, content="[3, 5, 7]")
}
