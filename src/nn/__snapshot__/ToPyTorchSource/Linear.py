import torch

_layer0 = torch.nn.Linear(1, 2)
_layer1 = torch.nn.ReLU()
_layer2 = torch.nn.Linear(2, 2)
_layer3 = torch.nn.ReLU()
_layer4 = torch.nn.Linear(2, 1)
_layer5 = torch.nn.Softmax(dim=-1)
_layer6 = torch.nn.Sequential(
    _layer0,
    _layer1,
    _layer2,
    _layer3,
    _layer4,
    _layer5,
)

with torch.no_grad():
    _layer0.weight = torch.nn.Parameter(torch.tensor([[-0.29557028390369,-0.8177422144361883]]).transpose(-2, -1))
    _layer0.bias = torch.nn.Parameter(torch.tensor([-0.8997844018869048,-0.1545487762608706]))
    _layer2.weight = torch.nn.Parameter(torch.tensor([[-0.39032679720623276,-0.06705152597716957],[0.46733088627300723,0.1554945481526122]]).transpose(-2, -1))
    _layer2.bias = torch.nn.Parameter(torch.tensor([0.3050002595219734,0.6775787971268634]))
    _layer4.weight = torch.nn.Parameter(torch.tensor([[0.5111579847372985],[-0.3668642715992912]]).transpose(-2, -1))
    _layer4.bias = torch.nn.Parameter(torch.tensor([0.1910001136989974]))

model = _layer6

__all__ = ["model"]