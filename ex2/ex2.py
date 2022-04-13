import torch
import torchvision 
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

torch.manual_seed(42)

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])

batch_size = 1000

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=True)

test_loader = DataLoader(test_dataset,
                        batch_size=batch_size,
                        shuffle=False)

alexnet = torchvision.models.alexnet(pretrained = True)
pretrained_alexnet = torchvision.models.alexnet(pretrained = True)

model = nn.Sequential(
    torchvision.transforms.Resize((63,63)),
    alexnet,
    nn.ReLU(),
    nn.Linear(1000, 10)
)

pretrained_model = nn.Sequential(
    torchvision.transforms.Resize((63,63)),
    pretrained_alexnet,
    nn.ReLU(),
    nn.Linear(1000, 10)
)

def accuracy(model, data_loader):
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()
        for _, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            pred = model(data)
            for i in range(len(labels)):
                pr = torch.argmax(pred[i], dim=-1)
                if pr == labels[i]:
                    correct += 1
                total += 1
        print(correct, total, correct/total)

def train(epochs, optimizer, loss_fn, train_loader, test_loader, model):
    for epoch in range(epochs):
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            pred = model(data)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(
                f'\rEpoch {epoch+1} [{i+1}/{len(train_loader)}] - Loss: {loss}',
                end=''
            )

        print('\n*************************************')
        print('Validation the model after epoch:', epoch)
        accuracy(model, val_loader)

model.to(device)
pretrained_model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
epochs = 20

print('Training AlexNet')
optimizer = torch.optim.Adam(model.parameters())
train(epochs, optimizer, loss_fn, train_loader, test_loader, model)
print('Test Accuracy')
accuracy(model, test_loader)

print('Pretrained AlexNet')
# freeze feature extraction (13 layers)
for i, param in enumerate(pretrained_model.parameters()):
    if i < 13:
        param.requires_grad = False

optimizer = torch.optim.Adam(pretrained_model.parameters())
train(4, optimizer, loss_fn, train_loader, test_loader, pretrained_model)

print('Test Accuracy')
accuracy(pretrained_model, test_loader)