test "cross_correlation" {
  let image = @torch.tensor(
    [
      [0.0, 1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0, 7.0],
      [8.0, 9.0, 10.0, 11.0],
      [12.0, 13.0, 14.0, 15.0],
    ],
  )
  let kernel = @torch.tensor(
    [[0.0, 1.0, -1.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
  )
  inspect!(@torch.correlate2d(image, kernel), content="[[15, 17], [23, 25]]")
}
