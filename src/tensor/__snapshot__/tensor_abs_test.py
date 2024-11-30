import torch
def test_abs_backward():
    a = torch.tensor([-2.075500011444092,1.0226000547409058,-0.08309999853372574,0.4805999994277954], requires_grad=True, dtype=torch.float32)
    b = torch.abs(a)
    b.sum().backward()
    assert torch.allclose(a.grad, torch.tensor([-1,1,-1,1], dtype=torch.float32))
