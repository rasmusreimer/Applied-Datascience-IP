import torch
from torch import nn
import torch.nn.functional as F

"""
A catelouge of different models with varying complexity of achitecture
"""

# 2layer nn
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, first_layer_size, second_layer_size):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, first_layer_size)
        self.fc2 = nn.Linear(first_layer_size, second_layer_size)
        self.fc3 = nn.Linear(second_layer_size, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# 3layer nn
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, first_layer_size, second_layer_size, third_layer_size):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, first_layer_size)
        self.fc2 = nn.Linear(first_layer_size, second_layer_size)
        self.fc3 = nn.Linear(second_layer_size, third_layer_size)
        self.fc4 = nn.Linear(third_layer_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# 4layer nn
class FourLayerNN(nn.Module):
    def __init__(self, input_size, first_layer_size, second_layer_size, third_layer_size, fourth_layer_size):
        super(FourLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, first_layer_size)
        self.fc2 = nn.Linear(first_layer_size, second_layer_size)
        self.fc3 = nn.Linear(second_layer_size, third_layer_size)
        self.fc4 = nn.Linear(third_layer_size, fourth_layer_size)
        self.fc5 = nn.Linear(fourth_layer_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x



# XGBoost classifier
from xgboost import XGBClassifier
class XGBoostModel:
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
