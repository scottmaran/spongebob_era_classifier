import matplotlib
matplotlib.use("Agg")

#from sklearn.preprocessing import LabelBinarizer - use for one-hot encoding for multiclass classification
#use labelEncoder for binary classification - changes strings to integers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# Reads image file paths and returns x and y lists
def read_paths():
    print("Loading images...")
    data = []
    labels = []

    #start with old images
    for image_file_name in os.listdir('spongebob_old_images'):
        try:
            path = os.path.join(os.getcwd(),'spongebob_old_images', image_file_name)
            image = cv2.imread(path).flatten()

            #image = cv2.imread(os.path.realpath(image_file_name)).flatten()
            # NO resizing done here
            data.append(image)
            labels.append('old')
        except:
            print('could not read: ', image_file_name)
            pass

    #do new images
    for image_file_name in os.listdir('spongebob_new_images'):
        try: 
            path = os.path.join(os.getcwd(),'spongebob_new_images', image_file_name)
            image = cv2.imread(path).flatten()
            #image = cv2.imread(os.path.realpath(image_file_name)).flatten()
            # NO resizing done here
            data.append(image)
            labels.append('new')
        except:
            print('could not read: ', image_file_name)
            pass

    #shuffle two lists with same order
    temp = list(zip(data, labels)) 
    random.shuffle(temp) 
    data, labels = zip(*temp) 
    return data, labels

data, labels = read_paths()

#SCALE raw pixel intensities & convert to numpy arrays
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

#change labels to integers
encoder = LabelEncoder()
trainY = encoder.fit_transform(trainY)
testY = encoder.transform(testY)


# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(32*32*3,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(1, activation="softmax"))
#in multiclass final output dim is len(encoder.classes_)

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 80
# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the neural network
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
#print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=encoder.classes_))
print(classification_report(testY, predictions, target_names=encoder.classes_))
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("output/simple_nn_plot.png")

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(output/simple_nn.model, save_format="h5")
f = open(output/simple_nn_lb.pickle, "wb")
f.write(pickle.dumps(encoder))
f.close()