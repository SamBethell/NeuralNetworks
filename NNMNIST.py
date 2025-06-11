import torch as t
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

t.manual_seed(0)

# loading dataset

train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = t.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = t.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1000,
                                          shuffle=False)

input_size = 784
hidden_size = 500
output_size = 10

class MyTorchLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(t.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(t.randn(out_features))
    def forward(self, x):
        return (x @ self.weights) + self.bias

class MyNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = MyTorchLinear(in_features, hidden_features)
        self.layer2 = MyTorchLinear(hidden_features, hidden_features)
        self.layer3 = MyTorchLinear(hidden_features, out_features)
    def forward(self, x):
        x = self.layer1(x)
        x = t.relu(x)
        x = self.layer2(x)
        x = t.relu(x)
        x = self.layer3(x)
        return x

net = MyNet(input_size, hidden_size, output_size)

class SGD():
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr
    def step(self):
        for param in self.parameters:
            param.data.add(-self.lr*param.grad)
    def zero(self):
        for param in self.parameters:
            param.grad.zero_()

def train():
    for images, labels in train_loader:
        images = images.reshape(-1, 28*28)
        labels = labels
        logits = net(images)
        #sgd = SGD(net.parameters(), 0.01)
        sgd = t.optim.SGD(net.parameters(), lr=0.1)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        sgd.step()
        sgd.zero_grad()

def eval(epoch):
    # Do one pass over the test data.
    # In the test phase, don't need to compute gradients (for memory efficiency)
    with t.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            #Convert image pixels to vector
            images = images.reshape(-1, 28*28).to()
            labels = labels.to()

            # Forward pass
            logits = net(images)

            # Compute total correct so far
            predicted = t.argmax(logits, -1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f'Test accuracy after {epoch+1} epochs: {100 * correct / total} %')

train()
for epoch in range(5):
    train()
    eval(epoch)
