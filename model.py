from statistics import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.layers import Dense
from keras.models import Sequential
from dataset_cleaning import cleanedDf as df

y = df["price"].values
x = df.drop("price",axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)



model = Sequential()

model.add(Dense(12,activation="relu")) #12 Neurons, Rectified Linear Units
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))


model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(x= x_train, y= y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)

#lossData = pd.DataFrame(model.history.history)
predictionArray = model.predict(x_test)
meanAbsoluteError = mean_absolute_error(y_test,predictionArray)
print(meanAbsoluteError)

plt.scatter(y_test,predictionArray)
plt.plot(y_test,y_test,"r-*")
plt.show()






