test "avg_pool2d" {
  let input = @tensor.tensor(
    [
      [
        [
          [1.0, 2.0, 3.0, 4.0],
          [2.0, 3.0, 4.0, 5.0],
          [3.0, 4.0, 5.0, 6.0],
          [4.0, 5.0, 6.0, 7.0],
        ],
      ],
    ],
  )
  inspect!(
    @tensor.avg_pool2d(input, kernel_size=(2, 2), stride=(2, 2)),
    content="[[[[2, 4], [4, 6]]]]",
  )
  inspect!(
    @tensor.avg_pool2d(input, kernel_size=(2, 2), stride=(1, 1)),
    content="[[[[2, 3, 4], [3, 4, 5], [4, 5, 6]]]]",
  )
}
