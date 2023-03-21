# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

First we can take the dataset based on one input value and some mathematical calculus output value.Next define the neural network model in three layers.First layer has six neurons and second layer has four neurons,third layer has one neuron.The neural network model takes the input and produces the actual output using regression.

## Neural Network Model
![network](https://user-images.githubusercontent.com/93427255/226674946-274a216c-bae1-4d11-b59e-51eef320d5e5.png)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

# Developed By: Gunaseelan G
# Register Number: 212220230031
```
from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('deeplearning').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'int'})
df = df.astype({'Output':'int'})

x = df[["Input"]] .values
y = df[["Output"]].values

scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

ai = Seq([
    Den(8,activation = 'relu',input_shape=[1]),
    Den(15,activation = 'relu'),
    Den(1),
])

ai.compile(optimizer = 'rmsprop',loss = 'mse')

ai.fit(x_train,y_train,epochs=2000)
ai.fit(x_train,y_train,epochs=2000)

loss_plot = pd.DataFrame(ai.history.history)
loss_plot.plot()

err = rmse()
preds = ai.predict(x_test)
err(y_test,preds)
x_n1 = [[30]]
x_n_n = scaler.transform(x_n1)
ai.predict(x_n_n)

```

## Dataset Information
![output1](https://user-images.githubusercontent.com/93427255/226675068-ca338ea2-3c7a-4307-9829-ee053871d1f0.png)


## OUTPUT



### Training Loss Vs Iteration Plot
![output2](https://user-images.githubusercontent.com/93427255/226675196-61b71965-3895-4713-8583-260de7b5f5b5.png)


### Test Data Root Mean Squared Error

![output3](https://user-images.githubusercontent.com/93427255/226675252-977dbb8c-d779-412a-8e31-b88b5a4b290b.png)


### New Sample Data Prediction

![output4](https://user-images.githubusercontent.com/93427255/226675295-7fde71ac-2d70-46d5-8457-ccff37db0aab.png)


## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
