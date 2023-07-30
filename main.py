import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from network import FusionNet
from dataset import get_dataset
from torch.utils.data import DataLoader,random_split
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

train_ratio = 0.8
batch_size = 5

dataset = get_dataset()
print("dataset:", dataset)
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloaders = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloaders = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = FusionNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decrease lr by a factor of 0.1 every 10 epochs



def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs=20):
    model.train()
    test_acc_list = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            labels = data[2]
            inputs = data[1]
            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Train Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        if (epoch + 1) % 5 == 0:
            test_acc = test_model(model, test_loader)
            test_acc_list.append(test_acc)

        # scheduler.step()

    return test_acc_list

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            labels = data[2]
            inputs = data[1]
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy  # return the accuracy so it can be recorded

test_acc_list = train_model(model, criterion, optimizer, scheduler, train_dataloaders, test_dataloaders, num_epochs=100)

# After the loop, plot the accuracy over time
plt.plot([5 * i for i in range(len(test_acc_list))], test_acc_list)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.show()
# Save the trained model
torch.save(model.state_dict(), "./saved_model.pt")
