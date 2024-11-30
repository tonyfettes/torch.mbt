pub fn vector_norm(
  input : @tensor.Tensor,
  ord~ : Int = 2,
  dim? : Int
) -> @tensor.Tensor {
  let input = match dim {
    Some(dim) => input.flatten()
    None => input
  }
  match ord {
    2 => input.pow(2).sum().sqrt()
    _ => ...
  }
}

// pub fn norm(
//   input : @tensor.Tensor,
//   ord? : NormOrd,
//   dim? : @shape.Shape
// ) -> @tensor.Tensor {
//   match (dim, ord) {
//     (Some(dim), Int(ord)) => {
//     }
//     (None, None) => {
//       let input = input.flatten()
//       input.pow(2).sum().sqrt()
//     }
//     (Some(Int(p)), None) => {
//       let p = p as Float
//       let norm = input.iter().map(|x| x.abs().powf(p)).sum().powf(1.0 / p)
//       @tensor.Tensor::new(norm)
//     }
//   }
// }
