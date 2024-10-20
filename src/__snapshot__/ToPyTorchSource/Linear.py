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
    _layer0.weight = torch.nn.Parameter(torch.tensor([[0.2319395440585279,-0.9572423457468497]]).transpose(-2, -1))
    _layer0.bias = torch.nn.Parameter(torch.tensor([-0.6499392928503491,-0.9692349476337747]))
    _layer2.weight = torch.nn.Parameter(torch.tensor([[-0.23773777815575792,-0.306780799224424],[-0.5823891499718423,0.48022393759696014]]).transpose(-2, -1))
    _layer2.bias = torch.nn.Parameter(torch.tensor([0.43009576963007823,0.2353595493072772]))
    _layer4.weight = torch.nn.Parameter(torch.tensor([[0.036133062528545534],[0.5297191538773458]]).transpose(-2, -1))
    _layer4.bias = torch.nn.Parameter(torch.tensor([-0.47865211700599963]))

model = _layer6

__all__ = ["model"]