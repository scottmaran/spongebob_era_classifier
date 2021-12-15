import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Reads image file paths and returns x and y lists
# old = 0, new = 1
def read_paths(random_seed):
    print("Loading images...")
    data = []
    labels = []

    #start with old images
    for image_file_name in os.listdir('spongebob_old_images_32x32'):
        try:
            path = os.path.join(os.getcwd(),'spongebob_old_images_32x32', image_file_name)
            image = cv2.imread(path).flatten()
            #print(type(image))

            #image = cv2.imread(os.path.realpath(image_file_name)).flatten()
            # NO resizing done here
            data.append(image)
            # append 
            #labels.append('old')
            labels.append(0)
        except:
            print('could not read: ', image_file_name)
            pass

    #do new images
    for image_file_name in os.listdir('spongebob_new_images_32x32'):
        try: 
            path = os.path.join(os.getcwd(),'spongebob_new_images_32x32', image_file_name)
            # scale color sizes
            image = cv2.imread(path).flatten() / 255.0
            #image = cv2.imread(os.path.realpath(image_file_name)).flatten()
            # NO resizing done here
            data.append(image)
            #labels.append('new')
            labels.append(1)
        except:
            print('could not read: ', image_file_name)
            pass

    data = np.stack(data,axis=0)
    data = data.reshape(-1, 32, 32, 3)

    #shuffle two lists with same order
    #temp = list(zip(data, labels)) 
    #np.random.seed(random_seed)
    #random.shuffle(temp) 
    #data, labels = zip(*temp) 
    return data, labels

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):

        model.train()  # put model to training mode
        X = X.to(device=device)  # move to device, e.g. GPU
        y = y.to(device=device)
        
        #compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred,y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            model.eval()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct

def train_part34(train_dataloader, model, loss_fn, optimizer, epochs, device):

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    acc_history = []
    loss_history = []
    for e in range(epochs):
        running_loss = 0
        for t, (x, y) in enumerate(train_dataloader):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = loss_fn(scores, y)

            #add loss from batch to running_loss
            running_loss += loss.item()

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            if(t%100 == 0):
                print('iteration ', t, '- loss: ', loss.item())

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            
            optimizer.step()
        # if(data_check):
        #     torch.save(model.state_dict(), f"models/{model_path}")
            
        
        #went through all examples
        #print(f"checking val accuracy for epoch {e}:")
        #acc = check_accuracy_part34(val_dataloader, model, device)
        #print(f"accuracy: {acc}")
        #if(scheduler is not None):
        #    scheduler.step(acc)
        # Takes too long to check training accuracy
        # print(f"checking train accuracy for epoch {e}:")
        #acc_train = check_accuracy_part34(train_dataloader, model, device)
        #print(f"accuracy: {acc_train}")

        #acc_history.append(acc)
        loss_history.append(running_loss/len(train_dataloader))

    return loss_history


'''
metric (str): e.g. 'accuracy', 'loss'
'''
def plot_history(history_list, metric, filename):

    # plotting
    plt.plot(list(range(1,len(history_list)+1)), history_list)
    plt.title("Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric}")
    plt.show()

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    plt.savefig(file_path)
