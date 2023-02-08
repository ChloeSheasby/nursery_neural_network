# the base of this comes from the github below and was adapted for my needs
# https://github.com/TheIndependentCode/Neural-Network

import numpy as np

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

list_of_inputs = [['usual', 'pretentious', 'great_pret'], 
                  ['proper', 'less_proper', 'improper', 'critical', 'very_crit'], 
                  ['complete', 'completed', 'incomplete', 'foster'], 
                  ['1', '2', '3', 'more'], 
                  ['convenient', 'less_conv', 'critical'], 
                  ['convenient', 'inconv'], 
                  ['non-prob', 'slightly_prob', 'problematic'], 
                  ['recommended', 'priority', 'not_recom']]

list_of_classes = ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']

def printLog(*args, **kwargs):
    print(*args, **kwargs)
    with open('output.txt','a') as file:
        print(*args, **kwargs, file=file)

def make_np_arrays(x, y):
    x = [np.hstack([np.isin(sublist, subapp, assume_unique=True).astype(int) for sublist in list_of_inputs]) for subapp in x]
    if y:
        y = [np.isin(list_of_classes, sublabel, assume_unique=True).astype(int) for sublabel in y]
    return x, y


def setup_data(x, y):
    x = np.array(x)
    x = x.reshape(x.shape[0], 27, 1)
    if y:
        y = np.array(y)
        y = y.reshape(y.shape[0], 5, 1)
    return x, y

def test_data(x_test, y_test, data, network, mode):
    correct = 0

    if(not skip or mode == "training"):
        for x, y, input in zip(x_test, y_test, data):
            output = predict(network, x)
            printLog('input:', input, '\npred:', list_of_classes[np.argmax(output)], '\ntrue:', list_of_classes[np.argmax(y)])
            if(list_of_classes[np.argmax(output)] == list_of_classes[np.argmax(y)]):
                correct += 1
                
        printLog(f"accuracy of {mode} data: {correct}/{len(data)} - {round((correct / len(data)) * 100, 5)}%")
    else:
        for x, input in zip(x_test, data):
            output = predict(network, x)
            printLog('input:', input, '\npred:', list_of_classes[np.argmax(output)])


printLog("Nursery Application Neural Network\n")
printLog("Hello, I have the ability to learn the patterns of deciding the outcomes of applications to this nursery.")
printLog("Please help me train by providing me with a training file. Each line should have 8 entries, all comma separated.")
fileName = input("Enter the path to a training file: ")
printLog("You provided me with this file:", fileName)

File = open(fileName,'r') # What we know!
nursery_applications = list(map(lambda x:x[:-1].rpartition(",")[0].split(','), File.readlines()))
File.close()

File = open(fileName,'r') # take labels from this
nursery_labels = list(map(lambda x: (x[:-1].rpartition(',')[-1]), File.readlines()))
File.close()   

printLog("\nThanks! Please give me a testing file in the same format at the training file.")
fileName = input("Enter the path to a testing file: ")
printLog("You provided me with this file:", fileName)

File2 = open(fileName,'r') # What we know!
nursery_applications_test = list(map(lambda x:x[:-1].split(','), File2.readlines()))
File2.close()

printLog("\nThanks! Now, if you would like to test my accuracy, please give me a test file that is labeled. If not, enter skip.")
fileName = input("Enter the path to a labeled testing file or 'skip': ")
skip = False
if(fileName.upper() == 'SKIP'):
    skip = True
    nursery_labels_test = []
else:
    printLog("You provided me with this file:", fileName)
    File2 = open(fileName,'r') # take labels from this
    nursery_labels_test = list(map(lambda x: (x[:-1].rpartition(',')[-1]), File2.readlines()))
    File2.close() 

printLog("\nGive me a bit of time to train and give you my results...\n")

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

printLog("I just finished training! I am going to test myself on my training data.")

test_data(x_train, y_train, nursery_applications, network, "training")

printLog("\nI just finished testing my training data! I will now give you your results.")

test_data(x_test, y_test, nursery_applications_test, network, "testing")