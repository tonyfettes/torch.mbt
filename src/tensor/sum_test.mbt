test "Tensor::sum/scalar" {
  let tensor = @tensor.tensor(1.0)
  let sum = tensor.sum()
  inspect!(sum.value, content="[1]")
  inspect!(sum.shape, content="[]")
  inspect!(sum.block, content="[]")
  inspect!(sum.graph, content="Sum(1, [[0]])")
  inspect!(sum.refcnt, content="0")
  inspect!(sum.grad, content="[0]")
  inspect!(sum, content="1")
}

test "Tensor::sum/vector" {
  let tensor = @tensor.tensor([0.0, 1.0, 2.0, 3.0])
  let sum = tensor.sum()
  inspect!(sum.value, content="[6]")
  inspect!(sum.shape, content="[]")
  inspect!(sum.block, content="[]")
  inspect!(sum.graph, content="Sum([0, 1, 2, 3], [[0, 1, 2, 3]])")
  inspect!(sum.refcnt, content="0")
  inspect!(sum.grad, content="[0]")
  inspect!(sum, content="6")
}

test "Tensor::sum/matrix" {
  let tensor = @tensor.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let sum = tensor.sum(dim=0)
  inspect!(sum.value, content="[3, 5, 7]")
  inspect!(sum.shape, content="[3]")
  inspect!(sum.block, content="[1]")
  inspect!(
    sum.graph,
    content="Sum([[0, 1, 2], [3, 4, 5]], [[0, 3], [1, 4], [2, 5]])",
  )
  inspect!(sum.refcnt, content="0")
  inspect!(sum.grad, content="[0, 0, 0]")
  inspect!(sum, content="[3, 5, 7]")
}

test "Tensor::sum/keep_dim" {
  let tensor = @tensor.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let sum = tensor.sum(dim=0, keep_dim=true)
  @json.inspect!(sum, content=[[3, 5, 7]])
}

test "Tensor::sum/dim=1" {
  let tensor = @tensor.tensor(
    [
      [0.0569, -0.2475, 0.0737, -0.3429],
      [-0.2993, 0.9138, 0.9337, -1.6864],
      [0.1132, 0.7892, -0.1003, 0.5688],
      [0.3637, -0.9906, -0.4752, -1.5197],
    ],
  )
  let sum = tensor.sum(dim=1)
  @json.inspect!(
    sum,
    content=[-0.45980000495910645,-0.13819992542266846,1.3708999156951904,-2.621799945831299],
  )
  let sum = tensor.sum(dim=1, keep_dim=true)
  @json.inspect!(
    sum,
    content=[
      [-0.45980000495910645],
      [-0.13819992542266846],
      [1.3708999156951904],
      [-2.621799945831299],
    ],
  )
}
