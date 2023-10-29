import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split


URL="https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"

RANDOM_NUMBER = 7

class NeuralNetwork(nn.Module): 

    def __init__(self, input_features=4, hidden_layer_1 = 8, hidden_layer_2 =9, output_features= 3, EPOCHS=100):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer_1) # fc1 means fully connected layer 1 
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.out = nn.Linear(hidden_layer_2, output_features)
        self.EPOCHS = EPOCHS

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x         


torch.manual_seed(RANDOM_NUMBER) # seed for randomization 

network = NeuralNetwork(EPOCHS=150)

data_frame = pd.read_csv(URL)

# change the word to a number to represent the classification of flower
data_frame['species'] = data_frame['species'].replace('setosa', 0)
data_frame['species'] = data_frame['species'].replace('versicolor', 1)
data_frame['species'] = data_frame['species'].replace('virginica', 2)


# Training Test Split 
x = data_frame.drop('species', axis=1) # X is everything but the species we are trying to predict, we are putting in out inputs of length width etc...
y = data_frame['species'] # dependent variable so we are making the prediction of y which is the flower species

x = x.values
y = y.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_NUMBER) # test on 20% of the data set

# using a float tensor because the data is in float type 
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

# long tensor is 64 bit integers
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# setting the criterion of model to measure the error. (This tells us how far off the predictions are from the actual data)
criterion = nn.CrossEntropyLoss()
# we also chose an optimizer which helps to improve the prediction. A common one is the Adam optimizer
# we are going to need to set a learning rate ( if the error does not go down after a bunch of itterations, then we will lower the learning rate)
# the lower the learning rate the longer it takes the model to learn, we can use a very small learning rate because we only have like 150 rows of data

optimizer = torch.optim.Adam(network.parameters(), lr=0.01) 

# epochs are the number of times the algorithm will scan our data esentially a run through our training data

losses = []

for i in range(network.EPOCHS):
    y_pred = network.forward(x_train)

    # measuring our loss 
    loss = criterion(y_pred, y_train) # comparing our predicited value vs our trained value 

    losses.append(loss.detach().numpy()) # this just keeps track of our losses so we should expect our losses to graudally go down through the EPOCHS


    # back propogation: take the error rate of forward propogation and feed it back through the network to fine tune the weights (this improves the predictions, which learns better)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # step forward again

plt.plot(range(network.EPOCHS), losses)
plt.ylabel("LOSSES/ERROR")
plt.xlabel("EPOCHS")
plt.show()

with torch.no_grad(): # turn of back propogation to evaluate the test data so that biases and weights are not being tuned
    y_eval = network.forward(x_test) # testing our model using the test set from x_test (esentially our 20% that was not trained)
    loss = criterion(y_eval, y_test) # compare our predicted values from the test set compared with out actual y test set
    print(loss)


correct = 0 
with torch.no_grad():
    for index, data in enumerate(x_test):
        y_eval = network.forward(data)

        prediction = y_eval.argmax().item()
        answer = y_test[index] # our actual answer will come from the test set
    

        # what network thinks our flower is 
        print(f"{index + 1} {str(y_eval)} \t Actual: {answer} \t\t\t Prediction: {prediction} ")
    

        if prediction == answer:
            correct += 1



print(f"Correct {correct} Accuracy: {round((correct/30) * 100, 2)}")


new_iris = torch.tensor([5.9,3.0,5.1,1.8]) # testing our model with a valid data point from set

with torch.no_grad():
    print(network.forward(new_iris).argmax().item()) # prediction should be 2 which is outputted

torch.save(network.state_dict(), "Aditya's Neural Network Iris Model.pt")

new_network = NeuralNetwork()
new_network.load_state_dict(torch.load("Aditya's Neural Network Iris Model.pt"))

print(new_network.eval())

"""Conclusion:
    This model reached an accuracy of 96.67%
    Too many EPOCHS result in a lower accuracy since the model is overfit for the data resulting in poor generalization on. 
    Too few EPOCHS doesnt allow the model to actually learn much leading to poor predictions. 
    
    """

