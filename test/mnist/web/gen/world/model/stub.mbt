// Generated by `wit-bindgen` 0.32.0.

let model : Ref[@mnist.Model] = Ref::new(@mnist.Model::new())

pub fn load(blob : String) -> Unit {
  model.val = try {
    blob |> @json.parse!() |> @json.from_json!()
  } catch {
    error => abort(Show::to_string(error))
  }
}

pub fn infer(input : FixedArray[Double]) -> FixedArray[Double] {
  model.val.forward(@torch.tensor(input)).value
}
