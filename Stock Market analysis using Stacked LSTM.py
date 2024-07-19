#!/usr/bin/env python
# coding: utf-8

# ### Stock Market Prediction And Forecasting Using Stacked LSTM

# In[1]:


### Keras and Tensorflow >2.0
get_ipython().system('pip install pandas_datareader')


# In[2]:


### Data Collection
import pandas_datareader as pdr
key="797d637c4cf0c76de2816da6b65b5ecb37f3fde7"
print('A')


# In[3]:


df = pdr.get_data_tiingo('AAPL', api_key=key)


# In[4]:


df.to_csv('AAPL.csv')


# In[6]:


get_ipython().system('ls')


# In[8]:


import pandas as pd


# In[406]:


df=pd.read_csv('AAPL.csv')


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df1=df.reset_index()['close']


# In[12]:


df1.shape


# In[13]:


df1


# In[14]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[291]:


### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 


# In[15]:


import numpy as np


# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[17]:


print(df1)
#an array is created in which values are between 0 and 1


# In[20]:


##splitting dataset into train and test split
#we have time series data, which means next data is dependent on previous data. so split like that
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[21]:


training_size,test_size


# In[22]:


train_data


# In[25]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[26]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[27]:


print(X_train.shape), print(y_train.shape)


# In[28]:


print(X_test.shape), print(ytest.shape)


# In[29]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[30]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[32]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[33]:


model.summary()


# In[34]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[35]:


import tensorflow as tf


# In[36]:


tf.__version__


# In[37]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[38]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[39]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[40]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[41]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[42]:


len(test_data)


# In[43]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[ ]:





# In[ ]:





# In[44]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[45]:


temp_input


# In[46]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[442]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[443]:


import matplotlib.pyplot as plt


# In[391]:


len(df1)


# In[392]:





# In[444]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[446]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[395]:


df3=scaler.inverse_transform(df3).tolist()


# In[396]:


plt.plot(df3)


# In[ ]:




