pub fn assert(result : Bool, message : String) -> Unit {
  if debug && not(result) {
    abort(message)
  }
}
