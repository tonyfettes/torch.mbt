test "sigmoid" {
  inspect!(@tensor.tensor(0.0).sigmoid(), content="0.5")
  inspect!(@tensor.tensor(-1.0).sigmoid(), content="0.2689414213699951")
  inspect!(@tensor.tensor(1.0).sigmoid(), content="0.7310585786300049")
  inspect!(
    @tensor.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]).sigmoid(),
    content="[0.01798620996209156, 0.11920292202211755, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823, 0.9820137900379085]",
  )
}
