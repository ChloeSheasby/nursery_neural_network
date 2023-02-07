# the base of this comes from the github below and was adapted for my needs
# https://github.com/TheIndependentCode/Neural-Network

import numpy as np

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

# Import the library
from tkinter import *
from tkinter import filedialog

list_of_inputs = [['usual', 'pretentious', 'great_pret'], 
                  ['proper', 'less_proper', 'improper', 'critical', 'very_crit'], 
                  ['complete', 'completed', 'incomplete', 'foster'], 
                  ['1', '2', '3', 'more'], 
                  ['convenient', 'less_conv', 'critical'], 
                  ['convenient', 'inconv'], 
                  ['non-prob', 'slightly_prob', 'problematic'], 
                  ['recommended', 'priority', 'not_recom']]

list_of_classes = ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']

def make_np_arrays(x, y):
    x = [np.hstack([np.isin(sublist, subapp, assume_unique=True).astype(int) for sublist in list_of_inputs]) for subapp in x]
    y = [np.isin(list_of_classes, sublabel, assume_unique=True).astype(int) for sublabel in y]
    return x, y


def setup_data(x, y):
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(x.shape[0], 27, 1)
    y = y.reshape(y.shape[0], 5, 1)
    return x, y

def test_data(x_test, y_test, data, network):
    correct = 0

    # testing the training data
    for x, y, input in zip(x_test, y_test, data):
        output = predict(network, x)
        print('input:', input, '\npred:', list_of_classes[np.argmax(output)], '\ntrue:', list_of_classes[np.argmax(y)])
        if(list_of_classes[np.argmax(output)] == list_of_classes[np.argmax(y)]):
            correct += 1

    print(f"accuracy of training data: {correct}/{len(data)} - {round((correct / len(data)) * 100, 5)}%")

# Create an instance of window
win=Tk()

win.title("Nursery Application Decider")

# Set the geometry of the window
win.geometry("700x300")

# Create a label
Label(win, text="Click the button to open a dialog", font='Arial 16 bold').pack(pady=15)

# Function to open a file in the system
def open_file():
   filepath = filedialog.askopenfilename(title="Give me a training data file:", filetypes=([("all files","*.*")]))
   file = open(filepath,'r')
   print(file.read())
   file.close()

# Create a button to trigger the dialog
button = Button(win, text="Open", command=open_file)
button.pack()

win.mainloop()

File = open('nursery-train.data','r') # What we know!
nursery_applications = list(map(lambda x:x[:-1].rpartition(",")[0].split(','), File.readlines()))
File.close()

File = open('nursery-train.data','r') # take labels from this
nursery_labels = list(map(lambda x: (x[:-1].rpartition(',')[-1]), File.readlines()))
File.close()   

File2 = open('nursery-test.data','r') # What we know!
nursery_applications_test = list(map(lambda x:x[:-1].split(','), File2.readlines()))
File2.close()

File2 = open('nursery-test-labeled.data','r') # take labels from this
nursery_labels_test = list(map(lambda x: (x[:-1].rpartition(',')[-1]), File2.readlines()))
File2.close() 

applications_train, labels_train = make_np_arrays(nursery_applications, nursery_labels)
applications_test, labels_test = make_np_arrays(nursery_applications_test, nursery_labels_test)


# training data
x_train, y_train = setup_data(applications_train, labels_train)
x_test, y_test = setup_data(applications_test, labels_test)

# neural network
network = [
    Dense(27, 20),
    Tanh(),
    Dense(20, 10),
    Tanh(),
    Dense(10, 5),
    Tanh()
]

# train
train(network, mse, mse_prime, x_train, y_train, epochs=100, learning_rate=0.1, verbose=False)

test_data(x_train, y_train, nursery_applications, network)
test_data(x_test, y_test, nursery_applications_test, network)