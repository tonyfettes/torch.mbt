import torch
def test_uniform_entropy():
    lower = torch.tensor(-3, requires_grad=True)
    upper = torch.tensor(5, requires_grad=True)
    normal = torch.distributions.Normal(lower, upper)
    entropy = normal.entropy()
    assert torch.allclose(entropy, torch.tensor(3.028376579284668))
    entropy.backward()
    assert torch.allclose(lower.grad, torch.tensor([0]))
    assert torch.allclose(upper.grad, torch.tensor([0.20000000298023224]))
