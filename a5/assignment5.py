from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def bar_graph(train, test, classes):
    counter = {}
    for i in range(len(train)):
        image, label = train[i]
        if classes[label.item()] not in counter:
            counter[classes[label.item()]] = 1
        else:
            counter[classes[label.item()]] += 1
    
    for i in range(len(test)):
        image, label = test[i]
        if classes[label.item()] not in counter:
            counter[classes[label.item()]] = 1
        else:
            counter[classes[label.item()]] += 1
    
    num_of_instances = []
    for c in classes:
        num_of_instances.append(counter[c])
    y_pos = np.arange(len(classes))
    plt.bar(y_pos,num_of_instances,width=0.4,align='center',alpha=0.5)
    plt.xticks(y_pos,classes)
    plt.title('number of instances of each class')
    # plt.show()

if __name__ == "__main__":
    train = datasets.FashionMNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test = datasets.FashionMNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    bar_graph(train, test, classes)