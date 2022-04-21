import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
import copy
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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

batch_size = 5000

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
        print('Validation test after epoch:', epoch+1)
        accuracy(model, test_loader)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        self.relu = nn.ReLU()
    
    def feature_extraction(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 28x28 > 14x14
        x = self.pool(self.relu(self.conv2(x))) # 14x14 > 7x7
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.fc1(x)
        return x

# Create an instance of the model
model = Model()

model.to(device)

epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
loss_fn = torch.nn.CrossEntropyLoss()

train(epochs, optimizer, loss_fn, train_loader_mnist, val_loader_mnist, model)

print('Test Accuracy:')
accuracy(model, test_loader_mnist)

image, labels = next(iter(val_loader_mnist))
image = image.to(device)
features = model.feature_extraction(image).cpu().detach().numpy()
print(features.shape)

cm = "tab10"

pca = PCA(n_components=2)
pca_values = pca.fit_transform(features, labels)

tsne = TSNE(n_components=2, perplexity=10)
tsne_values = tsne.fit_transform(features, labels)

fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15,10))
pca_x,pca_y = np.column_stack(pca_values)
ax1.scatter(pca_x, pca_y, c=labels, cmap=cm)
ax1.set_title('PCA 1')

tsne_x,tsne_y = np.column_stack(tsne_values)
ax2.scatter(tsne_x,tsne_y, c=labels, cmap=cm)
ax2.set_title('TSNE 1')

# Create an instance of the model
model2 = Model()

model2.to(device)

epochs = 20
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

train(epochs, optimizer, loss_fn, train_loader_mnist, val_loader_mnist, model2)

print('Test Accuracy:')
accuracy(model2, test_loader_mnist)

features2 = model2.feature_extraction(image).cpu().detach().numpy()

pca_values = pca.fit_transform(features2, labels)
pca_x,pca_y = np.column_stack(pca_values)
ax3.scatter(pca_x,pca_y,c=labels, cmap=cm)
ax3.set_title('PCA ' + str(epochs))

tsne_values = tsne.fit_transform(features2, labels)
tsne_x,tsne_y = np.column_stack(tsne_values)
ax4.scatter(tsne_x,tsne_y, c=labels, cmap=cm)
ax4.set_title('TSNE ' + str(epochs))

plt.savefig('test')