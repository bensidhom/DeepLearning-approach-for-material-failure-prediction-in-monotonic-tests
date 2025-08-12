
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
#from datetime import datetime

#Read the csv file
df = pd.read_excel(r'C:\all\data\monotonic_all_force_strain.xlsx')
print(df.head()) #7 columns, including the Date. 

train_value = 8575

# Today's forcast value depends on last 20-day values
window_size=100
# Only using "Open" feature for training
train_data=df.iloc[:(train_value+window_size)]
train_data.shape

train_data


#Variables for training
cols = list(df)[1:3]
#Date and volume columns are not used in training. 
print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']

#New dataframe with only training data - 5 columns
df_for_training =train_data[cols].astype(float)


df_for_training.plot.line()


#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 100  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 1])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

#In my case, trainX has a shape (12809, 14, 5). 
#12809 because we are looking back 14 days (12823 - 14 = 12809). 
#Remember that we cannot look back 14 days until we get to the 15th day. 
#Also, trainY has a shape (12809, 1). Our model only predicts a single value, but 
#it needs multiple variables (5 in my example) to make this prediction. 
#This is why we can only predict a single day after our training, the day after where our data ends.
#To predict more days in future, we need all the 5 variables which we do not have. 
#We need to predict all variables if we want to do that. 

# define the Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()


model.save('C:\\all\\data\\Force_stroke_100_1.keras')  # The file needs to end with the .keras extension
from keras.models import load_model

# Load the model from the .keras directory
model = load_model(r'C:\\all\\data\\Force_stroke_100_1.keras')



test_df = df.iloc[train_value:]
df_for_testing =test_df[cols].astype(float)
df_for_testing_scaled = scaler.transform(df_for_testing)
x_test = []

for i in range(n_past, len(df_for_testing_scaled) - n_future +1):
    x_test.append(df_for_testing_scaled[i - n_past:i, 0:df_for_testing.shape[1]])
#data = pd.concat((df_for_training_scaled, df_for_testing_scaled), axis=0)

x_test = np.array(x_test)

#Make prediction
prediction = model.predict(x_test) #shape = (n, 1) where n is the n_days_for_prediction
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,1]
plt.plot(y_pred_future,label='prediction')
plt.plot(a, label='true')
plt.legend()


a=list(test_df['force'][100:])

print('Mean Absolute Error: ', mean_absolute_error(a, y_pred_future))
print('Mean Squared Error: ', mean_squared_error(a, y_pred_future))
import pickle
scalerfile = r'C:\all\data\scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))


scaler = pickle.load(open(scalerfile, 'rb'))
#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
numbers = list(range(101, 151))

from numpy import newaxis
for i in range (1, len(x_test)-1):
    
  
    # We sorted descending
    curr_frame = x_test[i]

    copies2 = scaler.inverse_transform(curr_frame)[:,1]
   
 


    real1 = scaler.inverse_transform(x_test[i+1])[:,1][-1]

       
    
    future = []



    points_to_predict = 1
    for j in range(points_to_predict):
          # append the prediction to our empty future list
         prediction = model.predict(curr_frame[newaxis,:,:])
         prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
     
         y_pred_future = scaler.inverse_transform(prediction_copies)[:,1]
         future.append(y_pred_future)
          # insert our predicted point to our current frame
         curr_frame = np.insert(curr_frame, len(x_test[0]), future[-1], axis=0)
          # push the frame up one to make it progress into the future
         curr_frame = curr_frame[1:]
        

    # Reverse the original frame and the future frame


    plt.scatter(101,future[0], label=' prediction',color='black') 

    plt.scatter(101,real1, label='True',color='green')    

    #plt.plot(101,future,label='prediction',color='red')  

    plt.plot(copies2,label='History')
    plt.legend()
    plt.savefig(f"C:/all/data/force_stroke_100_1/fig_stroke_1{i}")

        # Display the plot and pause briefly to visualize
    plt.pause(0.2)  # Pauses for 0.5 seconds
    plt.title(f"Prediction at Step {i}")
    plt.xlabel('Window size')
    plt.ylabel('Force')
    #plt.savefig(f"C:/all/data/force_stroke_100_1/fig_stroke_1{i}")
# After the loop, keep the final plot on screen
plt.show()
