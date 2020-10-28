import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):  # feed-forward network

    def __init__(self):

        super().__init__()

        # how many neurons per layer? -> trial & error
        self.fc1 = nn.Linear(28*28, 64)  # fully connected layer
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):

        # activation function: rectified linear unit
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)  # log softmax

        return x
