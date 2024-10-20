import torch

layer0 = torch.nn.Linear(1, 2)
layer1 = torch.nn.ReLU()
layer2 = torch.nn.Linear(2, 2)
layer3 = torch.nn.ReLU()
layer4 = torch.nn.Linear(2, 1)
layer5 = torch.nn.Softmax(dim=-1)
layer6 = torch.nn.Sequential(
    layer0,
    layer1,
    layer2,
    layer3,
    layer4,
    layer5,
)

with torch.no_grad():
    layer0.weight = torch.nn.Parameter(torch.tensor([[0.2319395440585279,-0.9572423457468497]]))
    layer0.bias = torch.nn.Parameter(torch.tensor([-0.6499392928503491,-0.9692349476337747]))
    layer2.weight = torch.nn.Parameter(torch.tensor([[-0.23773777815575792,-0.306780799224424],[-0.5823891499718423,0.48022393759696014]]))
    layer2.bias = torch.nn.Parameter(torch.tensor([0.43009576963007823,0.2353595493072772]))
    layer4.weight = torch.nn.Parameter(torch.tensor([[0.036133062528545534],[0.5297191538773458]]))
    layer4.bias = torch.nn.Parameter(torch.tensor([-0.47865211700599963]))