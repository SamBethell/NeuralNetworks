import torch as t
import matplotlib.pyplot as plt
import torch.nn as nn



t.manual_seed(1) #Fix the random seed, so we always generate the same data.

N = 100
x_class_0 = 0.5*t.randn(N//2, 2) - 1
x_class_1 = t.randn(N//2, 2) + 1
X = t.cat([x_class_0, x_class_1], 0)
y = t.cat([t.zeros(N//2, 1), t.ones(N//2, 1)], 0)

plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

def loss(l):
    return -(y*t.nn.functional.logsigmoid(l) + (1-y)*t.nn.functional.logsigmoid(-l)).sum()

class MyLinear():
    def __init__(self, in_features, out_features):
        self.weights = t.randn(in_features, out_features, requires_grad=True)
        self.bias = t.randn(out_features, requires_grad=True)
    def __call__(self, x):
        return x@self.weights + self.bias
    def parameters(self):
        return [self.weights, self.bias]

class MyReLU():
    def __call__(self, x):
        return t.relu(x)
    def parameters(self):
        return []

class MySequential():
    def __init__(self, modules):
        self.modules = modules
    def __call__(self, x):
        for mod in self.modules:
            x = mod(x)
        return x
    def parameters(self):
        full_list = []
        for mod in self.modules:
            for param in mod:
                full_list.append(param)
        return full_list

input_features = 2
hidden_features = 100
output_features = 1
net = MySequential([
    MyLinear(input_features, hidden_features),
    MyReLU(),
    MyLinear(hidden_features, output_features)
])


#########################################################

class My_torchnn_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        # Calls the nn.Module __init__, which does some setup, so that
        # the convenient stuff around e.g. setting up parameters works.
        super().__init__()

        #Set up the parameters, by wrapping the initial tensors in `nn.Parameter`
        self.weights = nn.Parameter(t.randn(in_features, out_features))
        self.bias = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        return x @ self.weights + self.bias

class MyNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        self.layer1 = My_torchnn_Linear(input_features, hidden_features)
        self.layer2 = My_torchnn_Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.layer1(x)
        x = t.relu(x)
        x = self.layer2(x)
        return x

class MySGD():
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr
    def step(self):
        for param in self.parameters:
            param.data.add_(-self.lr*param.grad)
    def zero_grad(self):
        for param in self.parameters:
            param.grad.zero_()

learning_rate = 0.0001
mynet = MyNet(2, 100, 1)
sgd = MySGD(mynet.parameters(),learning_rate)
for i in range(100):
    L = loss(mynet(X))
    if 0 == i % 10:
        print(L.item())
    L.backward()
    sgd.step()
    sgd.zero_grad()
