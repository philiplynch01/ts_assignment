from experiments.hydra import Hydra, SparseScaler
from experiments.utils import get_cmj_data
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch import nn

x_train, y_train, x_test, y_test = get_cmj_data()

x_train = torch.from_numpy(x_train).float().unsqueeze(-2)
y_train = torch.from_numpy(y_train.astype(np.int32))
x_test = torch.from_numpy(x_test).float().unsqueeze(-2)
y_test = torch.from_numpy(y_test.astype(np.int32))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

device = torch.device("mps" if torch.backends.mps else "cpu")
print(device)

seed = 42
torch.manual_seed(seed)


transform = Hydra(x_train.shape[-1], k = 8, g = 64, seed = seed).to(device)

X_training_transform = transform(x_train)
X_test_transform = transform(x_test)

scaler = SparseScaler()

X_training_transform = scaler.fit_transform(X_training_transform)
X_test_transform = scaler.transform(X_test_transform)

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
classifier.fit(X_training_transform, y_train)

predictions = classifier.predict(X_test_transform)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
