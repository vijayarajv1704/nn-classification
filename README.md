# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/vinushcv/nn-classification/assets/113975318/95b22092-d6a4-41ac-8429-bf959435743e)


## DESIGN STEPS

### STEP 1:
Import the necessary packages & modules.

### STEP 2:
Load and read the dataset.

### STEP 3:
Perform pre processing and clean the dataset.

### STEP 4:
Normalize the values and split the values for x and y.

### STEP 5:
Build the deep learning model with appropriate layers and depth.

### STEP 6:
Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration.

### STEP 7:
Save the model using pickle.

### STEP 8:
Using the DL model predict for some random inputs.

## PROGRAM

### Name: vijayaraj v
### Register Number: 212222230174

```python
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv("customers.csv")


df.columns

df.shape

df.dtypes

df.isnull().sum()

df_cleaned=df.dropna(axis=0)

df_cleaned.isnull().sum()

df_cleaned.shape

df_cleaned.dtypes

df_cleaned["Gender"].unique()

df_cleaned["Ever_Married"].unique()

df_cleaned["Graduated"].unique()

df_cleaned["Profession"].unique()

df_cleaned["Spending_Score"].unique()

df_cleaned["Segmentation"].unique()

c_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]

enc=OrdinalEncoder(categories=c_list)

customer1=df_cleaned.copy()

customer1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']] = enc.fit_transform(customer1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

customer1.dtypes

le=LabelEncoder()

customer1["Segmentation"]=le.fit_transform(customer1["Segmentation"])

customer1.dtypes

customer1.describe()

x=customer1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1=customer1[["Segmentation"]].values

ohe=OneHotEncoder()

ohe.fit(y1)

y1.shape

y=ohe.transform(y1).toarray()

y.shape

y1[0]

y[0]

x.shape

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=50)

x_train[0]

x_train.shape

scaler=MinMaxScaler()

scaler.fit(x_train[:,2].reshape(-1,1))

x_train_scaled=np.copy(x_train)

x_test_scaled=np.copy(x_test)

x_train_scaled[:,2] = scaler.transform(x_train[:,2].reshape(-1,1)).reshape(-1)
x_test_scaled[:,2] = scaler.transform(x_test[:,2].reshape(-1,1)).reshape(-1)

ai_brain = Sequential([
  Dense(8,input_shape=(8,)),
  Dense(8,activation='relu'),
  Dense(8,activation='relu'),
  Dense(4,activation='softmax'),
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=2)

ai_brain.fit(x=x_train_scaled,y=y_train,
             epochs=20,batch_size=25,
             validation_data=(x_test_scaled,y_test),
             )

metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(ai_brain.predict(x_test_scaled), axis=1)

x_test_predictions.shape

y_test_truevalue = np.argmax(y_test,axis=1)

y_test_truevalue.shape


print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

ai_brain.save('customer_classification_model.h5')

with open('customer_data.pickle', 'wb') as fh:
  ai_brain = load_model('customer_classification_model.h5')
with open('customer_data.pickle', 'rb') as fh:
  x_single_prediction = np.argmax(ai_brain.predict(x_test_scaled[1:2,:]), axis=1)


print(x_single_prediction)



print(le.inverse_transform(x_single_prediction))




```

## Dataset Information

![image](https://github.com/vinushcv/nn-classification/assets/113975318/1faba58b-218e-4239-9f07-43e732230fa7)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/vinushcv/nn-classification/assets/113975318/487e8acd-30c5-431b-9597-6f020a4a1455)


### Classification Report

![image](https://github.com/vinushcv/nn-classification/assets/113975318/61ab97fe-ff3d-4f85-9794-d020e27b53ec)


### Confusion Matrix

![image](https://github.com/vinushcv/nn-classification/assets/113975318/2ab74aac-fecb-421e-a44d-0ea601f37852)



### New Sample Data Prediction

![image](https://github.com/vinushcv/nn-classification/assets/113975318/3a3b3aba-13a2-4b99-b120-f6def2a97b7d)


## RESULT
A neural network classification model is developed for the given dataset.


