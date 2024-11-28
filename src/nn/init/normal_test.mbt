test "normal_" {
  @distributions.seed(b"12345678223456783234567842345678")
  let tensor = @tensor.empty([2, 3])
  @init.normal_(tensor, mean=0.0, std=1.0)
  @json.inspect!(
    tensor.value,
    content=[
      1.2142412662506104, 0.7827171087265015, -2.1640777587890625, 1.1418954133987427,
      -1.653619408607483, 0.5077257752418518,
    ],
  )
}
