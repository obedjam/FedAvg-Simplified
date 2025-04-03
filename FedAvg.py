import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import numpy as np

# CNN Model Class
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train_local(train_data, val_data, model, epochs, l_rate, momentum, global_round, device, client_id):
    model.to(device)
    model.train()
    epoch_loss = []
    epoch_valid_loss = []
    batch_size = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=l_rate, momentum=momentum)
    criterion = nn.NLLLoss().to(device)

    for iter in range(epochs):
        batch_loss = []
        for images, labels in DataLoader(list(train_data), batch_size=batch_size, shuffle=True):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def test_inference(model, test_dataset, device):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(list(test_dataset), batch_size=128, shuffle=False)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss += criterion(outputs, labels).item()
        _, pred_labels = torch.max(outputs, 1)
        correct += torch.sum(pred_labels == labels).item()
        total += len(labels)
    return correct / total, loss

def split_local_data(train_dataset, num_clients):
    partition_size = [len(train_dataset) // num_clients] * num_clients
    remainder = len(train_dataset) % num_clients
    for i in range(remainder):
        partition_size[i] += 1
    local_model_data = random_split(train_dataset, partition_size)
    return [set(part) for part in local_model_data]

def get_dataset(num_clients):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("data/mnist/", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data/mnist/", train=False, download=True, transform=transform)
    local_data = split_local_data(train_dataset, num_clients)
    return test_dataset, local_data

def average_weights(local_weights):
    avg_weights = copy.deepcopy(local_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
    return avg_weights

if __name__ == '__main__':
    num_clients = 10
    num_classes = 10
    rounds = 5
    epochs = 10
    l_rate = 0.001
    momentum = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    global_model = CNN(num_classes).to(device)
    global_weights = global_model.state_dict()
    test_dataset, local_data = get_dataset(num_clients)
    local_models = [CNN(num_classes) for _ in range(num_clients)]

    for i in range(rounds):
        local_weights = []
        for k in range(num_clients):
            local_model = local_models[k]
            local_model.load_state_dict(global_weights)
            updated_weights, _ = train_local(local_data[k], None, local_model, epochs, l_rate, momentum, i, device, k)
            local_weights.append(updated_weights)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        accuracy, _ = test_inference(global_model, test_dataset, device)
        print(f"Round {i+1}: Global Accuracy = {accuracy*100:.2f}%")
