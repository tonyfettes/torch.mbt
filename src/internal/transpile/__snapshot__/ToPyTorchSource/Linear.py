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
    _layer0.weight = torch.nn.Parameter(torch.tensor([[-0.2955702841281891,-0.8177422285079956]]).transpose(-2, -1))
    _layer0.bias = torch.nn.Parameter(torch.tensor([-0.8997843861579895,-0.154548779129982]))
    _layer2.weight = torch.nn.Parameter(torch.tensor([[-0.3903267979621887,-0.06705152243375778],[0.4673308730125427,0.1554945409297943]]).transpose(-2, -1))
    _layer2.bias = torch.nn.Parameter(torch.tensor([0.3050002455711365,0.6775788068771362]))
    _layer4.weight = torch.nn.Parameter(torch.tensor([[0.5111579895019531],[-0.36686426401138306]]).transpose(-2, -1))
    _layer4.bias = torch.nn.Parameter(torch.tensor([0.19100011885166168]))

model = _layer6

__all__ = ["model"]