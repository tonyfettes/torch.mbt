test "sigmoid" {
  inspect!(@tensor.tensor(0.0).sigmoid(), content="0.5")
  inspect!(@tensor.tensor(-1.0).sigmoid(), content="0.2689414322376251")
  inspect!(@tensor.tensor(1.0).sigmoid(), content="0.7310585975646973")
  inspect!(
    @tensor.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]).sigmoid(),
    content="[0.01798621006309986, 0.11920291930437088, 0.2689414322376251, 0.5, 0.7310585975646973, 0.8807970285415649, 0.9820137619972229]",
  )
}
