test {
  inspect!(@dtype.of(true), content="Bool")
  inspect!(@dtype.of(1), content="Int")
  inspect!(@dtype.of((1.0 : Float)), content="Float")
  inspect!(@dtype.of(1.0), content="Double")
}
