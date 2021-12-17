import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from models import get_current_model
import helper_functions as help

data, labels = help.read_paths(random_seed=2430)

#SCALE raw pixel intensities & convert to numpy arrays
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ',device, ' device')

LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 17

tensor_trainX = torch.Tensor(trainX)
print(tensor_trainX.size())
tensor_trainX = tensor_trainX.permute(0,3,1,2)
tensor_trainY = torch.Tensor(trainY)
train_dataset = TensorDataset(tensor_trainX, tensor_trainY) # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

model = get_current_model()
model = model.to(device)
print('model', model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer_name = "Adam"

print("Beginning to train")
#(train_dataloader, model, loss_fn, optimzer, epochs, device)
loss_history, accuracy_history = help.train(train_dataloader, model, loss_fn, optimizer, EPOCHS, device)
model_filepath = 'saved_models/cnn_1.pt'
torch.save(model.state_dict(), model_filepath)

best_loss = round(min(loss_history), 3)
best_acc = round(max(accuracy_history), 3)
#print(f"Best training accuracy: {best_acc}")

acc_filename = f"plots/training_accuracy_cnn_1.png"
help.plot_history(accuracy_history, 'accuracy', acc_filename)

loss_filename = f"plots/training_loss_cnn_1.png"
help.plot_history(loss_history, 'loss', loss_filename)

print('Beginning to test')
tensor_testX = torch.Tensor(testX)
print(tensor_testX.size())
tensor_testX = tensor_testX.permute(0,3,1,2)
tensor_testY = torch.Tensor(testY)
test_dataset = TensorDataset(tensor_testX, tensor_testY)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
test_loss, acc_percentage = help.test_loop(test_dataloader, model, loss_fn, device)
print(f"Test Acc% = {acc_percentage}")