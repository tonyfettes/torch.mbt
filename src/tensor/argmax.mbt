pub fn argmax(input : Tensor, dim? : Int, keep_dim~ : Bool = false) -> Tensor {
  let (_, index) = max(input, dim?, keep_dim~)
  index
}
