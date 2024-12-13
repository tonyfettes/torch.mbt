///|
pub fn argmax(input : Tensor[Float], dim? : Int, keep_dim~ : Bool = false) -> Tensor[Float] {
  let (_, index) = max(input, dim?, keep_dim~)
  index
}
