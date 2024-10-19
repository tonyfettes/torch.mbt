test "AvgPool2d/Stride" {
  let input = @torch.tensor(
    [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
  )
  let avg_pool2d = @torch.AvgPool2d::new(2, stride=(1, 1))
  inspect!(avg_pool2d.forward(input), content="[[[[3, 4], [6, 7]]]]")
}

test "AvgPool2d/Backward" {
  let input = @torch.tensor(
    [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
  )
  let model = @torch.AvgPool2d::new(2, stride=(1, 1))
  let optimizer = @torch.SGD::new(Iter::singleton(input), learning_rate=1.0)
  let output = model.forward(input)
  @json.inspect!(output, content=[[[[2, 3], [5, 6]]]])
  optimizer.zero_grad()
  output.sum().backward()
  optimizer.step()
  @json.inspect!(
    input,
    content=[[[[-0.25, 0.5, 1.75], [2.5, 3, 4.5], [5.75, 6.5, 7.75]]]],
  )
}
