test "Shape" {
  @json.inspect!((3 as &@shape.Shape).to_fixed_array(), content=[3])
  @json.inspect!(([1, 2] as &@shape.Shape).to_fixed_array(), content=[1, 2])
  @json.inspect!((() as &@shape.Shape).to_fixed_array(), content=[])
}

test "Shape/iter" {
  @json.inspect!((3 as &@shape.Shape).iter().collect(), content=[3])
  @json.inspect!(([1, 2] as &@shape.Shape).iter().collect(), content=[1, 2])
  @json.inspect!((() as &@shape.Shape).iter().collect(), content=[])
}

test "Shape2d" {
  @json.inspect!((3 as &@shape.Shape2d).to_fixed_array(), content=[3, 3])
  @json.inspect!(((1, 2) as &@shape.Shape2d).to_fixed_array(), content=[1, 2])
  @json.inspect!((() as &@shape.Shape2d).to_fixed_array(), content=[0, 0])
  @json.inspect!((3 as &@shape.Shape2d).to_tuple(), content=[3, 3])
  @json.inspect!(((1, 2) as &@shape.Shape2d).to_tuple(), content=[1, 2])
  @json.inspect!((() as &@shape.Shape2d).to_tuple(), content=[0, 0])
}
