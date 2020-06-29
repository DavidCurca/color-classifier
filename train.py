import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from color import *

lookup = {
     0: 'red',
     1: 'green',
     2: 'blue',
     3: 'orange',
     4: 'yellow',
     5: 'pink',
     6: 'purple',
     7: 'brown',
     8: 'white',
     9: 'black'
}

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 6)
        self.fc4 = nn.Linear(6, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
dataset = ColorDataset()
dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
EPOCHS = int(input("Number of epochs?: "))
for epoch in range(EPOCHS):
    for data in dataset:
        X,y = data
        y = y.type(torch.LongTensor)
        net.zero_grad()
        output = net(X.view(-1, 3))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print("Loss: ", loss)

    
correct = 0
total = 0
with torch.no_grad():
    for data in dataset:
        X,y = data
        output = net(X.view(-1, 3))
        for idx, i in enumerate(output):
            if(torch.argmax(i) == y[idx]):
                correct += 1
            total += 1
print("Accuracy:{}".format(round(correct/total, 3)))
X = torch.tensor([1., 1., 0])
pred = torch.argmax(net(X.view(-1, 3)))
pred = pred.detach().numpy()
pred = np.append(pred, pred)
pred = pred[0]
print("Prediction: " + lookup[pred])
print("Expected  : yellow")
torch.save(net.state_dict(), "model.pth")
