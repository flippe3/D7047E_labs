import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

train_dataset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_dataset_mnist, val_dataset_mnist = random_split(train_dataset_mnist, [55000, 5000])

batch_size = 1000

train_loader_mnist = DataLoader(train_dataset_mnist,
                        batch_size=batch_size,
                        shuffle=True)
val_loader_mnist = DataLoader(val_dataset_mnist,
                        batch_size=batch_size,
                        shuffle=True)

test_loader_mnist = DataLoader(test_dataset_mnist,
                        batch_size=batch_size,
                        shuffle=False)

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

        print('\n')
        print('Validation test after epoch:', epoch)
        accuracy(model, test_loader)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 28x28 > 14x14
        x = self.pool(self.relu(self.conv2(x))) # 14x14 > 7x7
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
# Create an instance of the model
model = Model()

model.to(device)

epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

train(epochs, optimizer, loss_fn, train_loader_mnist, test_loader_mnist, model)

print('Test Accuracy:')
accuracy(model, test_loader_mnist)

model_svhn_feature = nn.Sequential(
    model,
    nn.ReLU(nn.Linear(1000, 10)),
)

model_svhn_fine = nn.Sequential(
    model,
    nn.ReLU(nn.Linear(1000, 10)),
)

model_svhn_feature.to(device)
model_svhn_fine.to(device)

transforms = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_dataset_svhn = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transforms)
test_dataset_svhn = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms)

train_dataset_svhn, val_dataset_svhn = random_split(train_dataset_svhn, [73257-10000, 10000])

train_loader_svhn = DataLoader(train_dataset_svhn,
                        batch_size=batch_size,
                        shuffle=True)                        

val_loader_svhn = DataLoader(train_dataset_svhn,
                        batch_size=batch_size,
                        shuffle=True)                        


test_loader_svhn = DataLoader(test_dataset_svhn,
                        batch_size=batch_size,
                        shuffle=False)          

print("\nMNIST CNN on SVHN")
accuracy(model, test_loader_svhn)

for i, param in enumerate(model_svhn_feature.parameters()):
    if i < 5:
        param.requires_grad = False

train(4, optimizer, loss_fn, train_loader_svhn, val_loader_svhn, model_svhn_feature)

print("\nMNIST ON CNN AFTER FEATURE-EXTRACTION")
accuracy(model_svhn_feature, test_loader_svhn)

train(epochs, optimizer, loss_fn, train_loader_svhn, val_loader_svhn, model_svhn_fine)

print("\nMNIST ON CNN AFTER FINE-TUNING")
accuracy(model_svhn_fine, test_loader_svhn)

