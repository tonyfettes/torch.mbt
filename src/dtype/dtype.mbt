///|
enum Type {
  Bool
  Int
  Int64
  Float
  Double
} derive(Show)

///|
trait Dtype {
  dtype() -> Type
}

///|
pub impl Dtype for Bool with dtype() -> Type { Type::Bool }

///|
pub impl Dtype for Int with dtype() -> Type { Type::Int }

///|
pub impl Dtype for Int64 with dtype() -> Type { Type::Int64 }

///|
pub impl Dtype for Float with dtype() -> Type { Type::Float }

///|
pub impl Dtype for Double with dtype() -> Type { Type::Double }

///|
pub fn of[T : Dtype](_value : T) -> Type {
  T::dtype()
}
