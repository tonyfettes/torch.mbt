import torch
def test_uniform_entropy():
    lower = torch.tensor(-3, requires_grad=True)
    upper = torch.tensor(5, requires_grad=True)
    uniform = torch.distributions.Uniform(lower, upper)
    entropy = uniform.entropy()
    assert torch.allclose(entropy, torch.tensor(2.079441547393799))
    entropy.backward()
    assert torch.allclose(lower.grad, torch.tensor([-0.125]))
    assert torch.allclose(upper.grad, torch.tensor([0.125]))
