
##########################
##Part 1- Preprocessing##
########################

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Google_Stock_Price_Train.csv')
# in rnn the input is current stock price and output stock price
# Here taking open pice as our dataset
dataset = dataset.iloc[:, 1:2].values

# Feture scaling
# we can use standardization or normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
dataset = sc.fit_transform(dataset)

# now geting the feature(X) and label(y)
X_train = dataset[0:1257, :]
y_train = dataset[1:1258]

# Reshaping the input i.e only th X-train
# Chainging 2-Ddimension of input to 3-D
X_train = np.reshape(X_train, (1257, 1, 1))


##########################
### Building the RNN ####
#########################

# importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Sequential to initialize the rnn
# dense to create output layers
#LSTM ie Long Term Memmory

# initialize the RNN
# since stock price is regular no, using regerssion here
regressor = Sequential()

# Now creating layers, input hidden and output
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting the RNN to training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

#####################################################
## Part-3: MAking prediction $ visualising result ##
###################################################
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
X_test = test_set.iloc[:, 1:2]

# scaling the X_test
input = sc.transform(X_test)

# before predictiong , converting in 3-D
input = np.reshape(input, (20, 1, 1))

# now predicting the next price
y_pred = regressor.predict(input)

# since we are getting the scaled output so, using inverse_transform()
y_pred = sc.inverse_transform(y_pred)

# Visualization of result
plt.plot(X_test, color ='red', label = "Real Google Price")
plt.plot(y_pred, color = 'blue', label = "Predicted Google Price")
plt.title("Price_Prediction")
plt.x_label("Time")
plt.y_label("Stock-Price-Google")
plt.legend()
plt.show()





