test "tensor" {
  @json.inspect!(@torch.tensor([1.0, 2.0, 3.0]), content=[1, 2, 3])
}
