pub impl[A : Module, B : Module] Module for (A, B) with forward(
  self : (A, B),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input |> forward_(self.0) |> forward_(self.1)
}

pub impl[A : Module, B : Module] Module for (A, B) with parameters(
  self : (A, B)
) -> Iter[@tensor.Tensor] {
  self.0.parameters() |> Iter::concat(self.1.parameters())
}

pub impl[A : Module, B : Module, C : Module] Module for (A, B, C) with forward(
  self : (A, B, C),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input |> forward_(self.0) |> forward_(self.1) |> forward_(self.2)
}

pub impl[A : Module, B : Module, C : Module] Module for (A, B, C) with parameters(
  self : (A, B, C)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module] Module for (A, B, C, D) with forward(
  self : (A, B, C, D),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
}

pub impl[A : Module, B : Module, C : Module, D : Module] Module for (A, B, C, D) with parameters(
  self : (A, B, C, D)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module] Module for (
  A,
  B,
  C,
  D,
  E,
) with forward(self : (A, B, C, D, E), input : @tensor.Tensor) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module] Module for (
  A,
  B,
  C,
  D,
  E,
) with parameters(self : (A, B, C, D, E)) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
) with forward(self : (A, B, C, D, E, F), input : @tensor.Tensor) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
) with parameters(self : (A, B, C, D, E, F)) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
) with forward(self : (A, B, C, D, E, F, G), input : @tensor.Tensor) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
) with parameters(self : (A, B, C, D, E, F, G)) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
) with forward(self : (A, B, C, D, E, F, G, H), input : @tensor.Tensor) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
) with parameters(self : (A, B, C, D, E, F, G, H)) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
) with forward(self : (A, B, C, D, E, F, G, H, I), input : @tensor.Tensor) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
) with parameters(self : (A, B, C, D, E, F, G, H, I)) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
) with forward(self : (A, B, C, D, E, F, G, H, I, J), input : @tensor.Tensor) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
) with parameters(self : (A, B, C, D, E, F, G, H, I, J)) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
) with forward(self : (A, B, C, D, E, F, G, H, I, J, K), input : @tensor.Tensor) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
) with parameters(self : (A, B, C, D, E, F, G, H, I, J, K)) -> Iter[
  @tensor.Tensor,
] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
) with parameters(self : (A, B, C, D, E, F, G, H, I, J, K, L)) -> Iter[
  @tensor.Tensor,
] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
  |> forward_(self.12)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
) with parameters(self : (A, B, C, D, E, F, G, H, I, J, K, L, M)) -> Iter[
  @tensor.Tensor,
] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
  |> Iter::concat(self.12.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
  |> forward_(self.12)
  |> forward_(self.13)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
) with parameters(self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N)) -> Iter[
  @tensor.Tensor,
] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
  |> Iter::concat(self.12.parameters())
  |> Iter::concat(self.13.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
  |> forward_(self.12)
  |> forward_(self.13)
  |> forward_(self.14)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
) with parameters(self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)) -> Iter[
  @tensor.Tensor,
] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
  |> Iter::concat(self.12.parameters())
  |> Iter::concat(self.13.parameters())
  |> Iter::concat(self.14.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module, P : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
  P,
) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
  |> forward_(self.12)
  |> forward_(self.13)
  |> forward_(self.14)
  |> forward_(self.15)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module, P : Module] Module for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
  P,
) with parameters(self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)) -> Iter[
  @tensor.Tensor,
] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
  |> Iter::concat(self.12.parameters())
  |> Iter::concat(self.13.parameters())
  |> Iter::concat(self.14.parameters())
  |> Iter::concat(self.15.parameters())
}
