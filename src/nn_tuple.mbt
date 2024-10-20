pub impl[A : Module, B : Module] Module for (A, B) with forward(
  self : (A, B),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
}

pub impl[A : Module, B : Module] Module for (A, B) with parameters(
  self : (A, B)
) -> Iter[Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
}

pub impl[A : Module, B : Module, C : Module] Module for (A, B, C) with forward(
  self : (A, B, C),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
}

pub impl[A : Module, B : Module, C : Module] Module for (A, B, C) with parameters(
  self : (A, B, C)
) -> Iter[Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module] Module for (A, B, C, D) with forward(
  self : (A, B, C, D),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
}

pub impl[A : Module, B : Module, C : Module, D : Module] Module for (A, B, C, D) with parameters(
  self : (A, B, C, D)
) -> Iter[Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module] Module for (A, B, C, D, E) with forward(
  self : (A, B, C, D, E),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module] Module for (A, B, C, D, E) with parameters(
  self : (A, B, C, D, E)
) -> Iter[Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module] Module for (A, B, C, D, E, F) with forward(
  self : (A, B, C, D, E, F),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module] Module for (A, B, C, D, E, F) with parameters(
  self : (A, B, C, D, E, F)
) -> Iter[Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module] Module for (A, B, C, D, E, F, G) with forward(
  self : (A, B, C, D, E, F, G),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module] Module for (A, B, C, D, E, F, G) with parameters(
  self : (A, B, C, D, E, F, G)
) -> Iter[Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module] Module for (A, B, C, D, E, F, G, H) with forward(
  self : (A, B, C, D, E, F, G, H),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module] Module for (A, B, C, D, E, F, G, H) with parameters(
  self : (A, B, C, D, E, F, G, H)
) -> Iter[Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module] Module for (A, B, C, D, E, F, G, H, I) with forward(
  self : (A, B, C, D, E, F, G, H, I),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
  |> forward(self.8)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module] Module for (A, B, C, D, E, F, G, H, I) with parameters(
  self : (A, B, C, D, E, F, G, H, I)
) -> Iter[Tensor] {
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module] Module for (A, B, C, D, E, F, G, H, I, J) with forward(
  self : (A, B, C, D, E, F, G, H, I, J),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
  |> forward(self.8)
  |> forward(self.9)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module] Module for (A, B, C, D, E, F, G, H, I, J) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J)
) -> Iter[Tensor] {
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module] Module for (A, B, C, D, E, F, G, H, I, J, K) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
  |> forward(self.8)
  |> forward(self.9)
  |> forward(self.10)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module] Module for (A, B, C, D, E, F, G, H, I, J, K) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K)
) -> Iter[Tensor] {
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
  |> forward(self.8)
  |> forward(self.9)
  |> forward(self.10)
  |> forward(self.11)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L)
) -> Iter[Tensor] {
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
  |> forward(self.8)
  |> forward(self.9)
  |> forward(self.10)
  |> forward(self.11)
  |> forward(self.12)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M)
) -> Iter[Tensor] {
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
  |> forward(self.8)
  |> forward(self.9)
  |> forward(self.10)
  |> forward(self.11)
  |> forward(self.12)
  |> forward(self.13)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
) -> Iter[Tensor] {
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
  |> forward(self.8)
  |> forward(self.9)
  |> forward(self.10)
  |> forward(self.11)
  |> forward(self.12)
  |> forward(self.13)
  |> forward(self.14)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
) -> Iter[Tensor] {
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module, P : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
  input : Tensor
) -> Tensor {
  input
  |> forward(self.0)
  |> forward(self.1)
  |> forward(self.2)
  |> forward(self.3)
  |> forward(self.4)
  |> forward(self.5)
  |> forward(self.6)
  |> forward(self.7)
  |> forward(self.8)
  |> forward(self.9)
  |> forward(self.10)
  |> forward(self.11)
  |> forward(self.12)
  |> forward(self.13)
  |> forward(self.14)
  |> forward(self.15)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module, P : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)
) -> Iter[Tensor] {
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