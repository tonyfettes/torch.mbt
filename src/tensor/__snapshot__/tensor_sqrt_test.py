import torch
def test_sqrt_backward():
    a = torch.tensor([1.0226000547409058,0.08309999853372574,0.4805999994277954], requires_grad=True, dtype=torch.float32)
    b = torch.sqrt(a)
    b.sum().backward()
    assert torch.allclose(a.grad, torch.tensor([0.49444398283958435,1.7344807386398315,0.7212371826171875], dtype=torch.float32))
