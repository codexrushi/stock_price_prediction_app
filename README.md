# stock_price_prediction_app
the stock price prediction based on LSTM model using streamlit we create a web application.
# Required libraries:
1) numpy
2) pandas
3) matplotlib
4) keras
5) streamlit
# Required medium open source:
1)jupyter notebook
2)spyder
# Source code:

Created on Sat May 28 14:39:09 2022
# @author: RUSHIKESH
Created on Sat May 28 14:39:09 2022
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start=2010-1-1
end=2021-1-1


st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')

df=data.DataReader(user_input,'yahoo',start,end)

#describing data 
st.subheader('Data from 2010-2021')
st.write(df.describe())

#visulization
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

#splitting data into training and testing 
#70% data is training data and 30% data is testing data
data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing= pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
#print(data_training.shape)
#print(data_testing.shape)


#Now we have to scaling down the data between 0 and 1
#for the stat LSTM model we have to scale down the data, we cant provide this data directly
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



#load my LSTM model which we have created and save into the file 

model=load_model('keras_model.h5')

#testing part start


past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test=np.array(x_test),np.array(y_test)  
y_predicted= model.predict(x_test)  
scaler=scaler.scale_
scale_factor= 1/scaler[0]
y_predicted= y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


 
