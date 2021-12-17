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
# currently size needs to be either 32 or 128, as those are the only datasets stored in these repo
def read_paths(random_seed, size=32):
    print("Loading images...")
    data = []
    labels = []
    print(f"Size = {size}")

    old_path = f'spongebob_old_images_{size}x{size}'
    new_path = f'spongebob_new_images_{size}x{size}'

    #start with old images
    for image_file_name in os.listdir(old_path):
        try:
            path = os.path.join(os.getcwd(),old_path, image_file_name)
            image = cv2.imread(path).flatten()
            # NO resizing or scaling done here
            data.append(image)
            #labels.append('old')
            labels.append(0)
        except:
            print('could not read: ', image_file_name)
            pass

    #do new images
    for image_file_name in os.listdir(new_path):
        try: 
            path = os.path.join(os.getcwd(),new_path, image_file_name)
            image = cv2.imread(path).flatten()
            # NO resizing done here
            data.append(image)
            #labels.append('new')
            labels.append(1)
        except:
            print('could not read: ', image_file_name)
            pass

    data = np.stack(data,axis=0)
    data = data.reshape(-1, size, size, 3)

    #shuffle two lists with same order
    #temp = list(zip(data, labels)) 
    #np.random.seed(random_seed)
    #random.shuffle(temp) 
    #data, labels = zip(*temp) 
    return data, labels

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

def train(train_dataloader, model, loss_fn, optimizer, epochs, device, save_while_training=False, model_filepath='NA'):

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    num_batches = len(train_dataloader)
    num_training_samples = len(train_dataloader.dataset)

    acc_history = []
    loss_history = []
    for e in range(epochs):
        running_loss = 0
        correct = 0
        for t, (x, y) in enumerate(train_dataloader):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = loss_fn(scores, y)

            correct += (scores.argmax(1) == y).type(torch.float).sum().item()

            #add loss from batch to running_loss
            running_loss += loss.item()

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            if(t%30 == 0):
                print(f"epoch {e+1}: iter {t} - loss = {loss.item()}")

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
        
        acc = correct/num_training_samples
        print(f"Epoch {e+1} running loss = {running_loss/num_batches}, accuracy = {acc}")

        if save_while_training:
             torch.save(model.state_dict(), model_filepath)
            
        acc_history.append(acc)
        loss_history.append(running_loss/num_batches)

    return loss_history, acc_history


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

'''
Function to display an image given a dataloader. Assuming the dataloader is returns a batch of images, 
batch index is the index of the element you want to visualize. The function only displays one image from
the batch
'''
def display_image(dataloader, batch_index=0, verbose=False):
    print('display an image')
    train_features, train_labels = next(iter(dataloader))
    if verbose:
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[batch_index].squeeze().cpu()
    # need to flip it b/c opencv reads in images as BGR, matplot reads in as RGB
    img = torch.flip(img,dims=(0,))
    label = train_labels[batch_index]
    plot_img = torch.moveaxis(img,(0,1,2),(-1,0,1))
    if verbose:
        print('plot image shape', plot_img.shape)
    plt.imshow(plot_img)
    plt.show()
    if verbose:
        print(f"Label: {label}")

'''
Function to train model for a single batch. Not used
'''
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
