import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import data_loader as dl
from PIL import Image
from model import ResNet, BasicBlock, ModInception
from torch.utils.data import DataLoader, Dataset, random_split

epochs = 85
Costs = []
Accuracy = []
batch_size = 32
learning_rate = 0.0002
lmbda = 0
data_path = {'train' : "data/train/", 'test' : "data/test/"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_accuracy(output, label, batch_size):
    
    label = label.detach().cpu().numpy().squeeze()
    output = output.detach().cpu()
    _, indices = torch.max(output, dim=1)
    output = torch.zeros_like(output)
    itr = iter(indices)
    for i in range(output.shape[0]):
        output[i, int(next(itr))] = 1
    
    label = torch.tensor(np.eye(10)[label]).float()
    diff = torch.sum(torch.abs(output - label))/(2*output.shape[0])
    acc = 100 - (100 * diff)
    
    return acc


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


data = dl.TamilVowelConsonantDataset(data_path, transform = transform, train = True)
train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_data, validation_data = random_split(data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)


test_data = dl.TamilVowelConsonantDataset(data_path, transform = transform, train = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=False)


deepNet = ResNet(BasicBlock, ModInception).to(device)
cnn_ce_loss = nn.CrossEntropyLoss()
cnnet_optim = optim.Adam(deepNet.parameters(), lr = learning_rate, weight_decay=lmbda)


deepNet = deepNet.train()

for epoch in range(epochs):
    acc = 0
    count = 0
    for i, batch in enumerate(train_loader):

        print("started")
        break
        
        count += 1
        images, label = batch
        images = images.to(device)
        label['Vowel'] = label['Vowel'].to(device).long()
        label['Consonant'] = label['Consonant'].to(device).long()
        
        cnnet_optim.zero_grad()
        out_1, out_2 = deepNet(images)
        loss_1 = cnn_ce_loss(out_1, label['Vowel'])
        loss_2 = cnn_ce_loss(out_2, label['Consonant'])
        loss = loss_1 + loss_2
        loss.backward()
        cnnet_optim.step()
        cost = loss.item()
        
        acc1 = get_accuracy(out_1, label['Vowel'], images.shape[0])
        acc2 = get_accuracy(out_2, label['Consonant'], images.shape[0])
        acc = acc + (acc1 + acc2)/2
    break
    
    Accuracy.append(acc/(10*count))
    Costs.append(cost)    
    print("Epoch [{}/{}], Loss : {}, Accuracy : {}, acc_1 : {}, acc_2 : {}".format((epoch+1), epochs,
                                                                                   round(float(cost), 2),
                                                                                   round(float(Accuracy[-1].cpu()), 2),
                                                                                   round(float(acc1.cpu()), 2),
                                                                                   round(float(acc2.cpu()), 2)))

plt.title("Loss and Accuracy with iterations")
plt.plot(Costs, label = 'Cost')
plt.plot(Accuracy, label = 'Accuracy')
plt.xlabel("Iterations")
plt.ylabel("Loss and Accuracy")
plt.legend()
plt.show()

count = 0
acc = 0

deepNet = deepNet.eval()

for i, batch in enumerate(train_loader):
    
    count += 1
    images, label = batch
    images = images.to(device)
    label['Vowel'] = label['Vowel'].to(device).long()
    label['Consonant'] = label['Consonant'].to(device).long()

    out_1, out_2 = deepNet(images)
    out_1, out_2 = F.log_softmax(out_1, dim = 1), F.log_softmax(out_2, dim = 1)
    
    acc1 = get_accuracy(out_1, label['Vowel'], images.shape[0])
    acc2 = get_accuracy(out_2, label['Consonant'], images.shape[0])
    acc = acc + (acc1 + acc2)/2

print("Train Accuracy : {}".format(acc/count))
