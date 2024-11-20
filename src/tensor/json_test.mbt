test "Tensor::to_json" {
  let tensor = @tensor.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}

test "Tensor::from_json" {
  let json : Json = [[0, 1, 2], [3, 4, 5]]
  let tensor : @tensor.Tensor = json |> @json.from_json!()
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}
