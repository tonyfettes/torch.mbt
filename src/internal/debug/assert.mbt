pub fn assert(result : Bool, message : String) -> Unit {
  if DEBUG && not(result) {
    abort(message)
  }
}
