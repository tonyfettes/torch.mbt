test "AvgPool2d/Stride" {
  let input = @torch.tensor(
    [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
  )
  let avg_pool2d = @torch.AvgPool2d::new(2, stride=(1, 1))
  inspect!(avg_pool2d.forward(input), content="[[[[3, 4], [6, 7]]]]")
}
