let no_grad : Ref[Bool] = Ref::new(false)

pub fn with_no_grad[X](f : () -> X) -> X {
  no_grad.protect(true, f)
}

pub fn with_no_grad_error[X](f : () -> X!) -> X! {
  let old = no_grad.val
  no_grad.val = true
  let r = f!()
  no_grad.val = old
  r
}
