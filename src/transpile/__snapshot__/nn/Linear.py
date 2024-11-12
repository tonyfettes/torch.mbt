import torch

_layer0 = torch.nn.Linear(1, 2)
with torch.no_grad():
    _layer0.weight = torch.nn.Parameter(torch.tensor([[-0.2955702543258667,-0.8177422285079956]]).transpose(-2, -1))
    _layer0.bias = torch.nn.Parameter(torch.tensor([-0.8997843861579895,-0.1545487642288208]))
_layer1 = torch.nn.ReLU()
_layer2 = torch.nn.Linear(2, 2)
with torch.no_grad():
    _layer2.weight = torch.nn.Parameter(torch.tensor([[-0.3903267979621887,-0.06705152988433838],[0.4673308730125427,0.1554945707321167]]).transpose(-2, -1))
    _layer2.bias = torch.nn.Parameter(torch.tensor([0.3050002455711365,0.6775787472724915]))
_layer3 = torch.nn.ReLU()
_layer4 = torch.nn.Linear(2, 1)
with torch.no_grad():
    _layer4.weight = torch.nn.Parameter(torch.tensor([[0.5111579298973083],[-0.36686426401138306]]).transpose(-2, -1))
    _layer4.bias = torch.nn.Parameter(torch.tensor([0.1910000443458557]))
_layer5 = torch.nn.Softmax(dim=-1)
model = torch.nn.Sequential(
    _layer0,
    _layer1,
    _layer2,
    _layer3,
    _layer4,
    _layer5,
)

__all__ = ["model"]
