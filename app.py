import streamlit as st
import numpy as np
from nsepy import get_history
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.set_printoptions(suppress=True)
st.title('Stock Market Prediction using LSTM')
col1,col2=st.columns(2)
startDate=(col1.date_input('Enter Start Date'))
endDate=(col2.date_input('Enter End Date'))

symbol=st.text_input('Enter Stock Symbol')

if st.button('Get Data'):
    StockData=get_history(symbol=symbol,start=startDate,end=endDate)
    print(StockData.shape)
    print(StockData.columns)
    StockData['TradeDate']=StockData.index
    fig=plt.figure(figsize=(20,6))
    plt.plot(StockData['TradeDate'],StockData['Close'])
    plt.title('Stock Prices Vs Date')
    plt.xlabel('TradeDate')
    plt.ylabel('Stock Price')
    st.pyplot(fig)

    FullData=StockData[['Close']].values
    st.header('Before Normalization')
    st.write(FullData[0:5])

    sc=MinMaxScaler()
    DataScaler=sc.fit(FullData)
    X=DataScaler.transform(FullData)
    st.header('After Normalization')
    st.write(X[0:5])

    X_samples=list()
    Y_samples=list()
    NumberOfRows=len(X)
    TimeSteps=10

    for i in range(TimeSteps, NumberOfRows,1):
        X_sample=X[i-TimeSteps:i]
        Y_sample=X[i]
        X_samples.append(X_sample)
        Y_samples.append(Y_sample)
    
    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1],1)

    Y_data=np.array(Y_samples)
    Y_data=Y_data.reshape(Y_data.shape[0],1)

    st.header('Data Shapes for LSTM')
    col1,col2=st.columns(2)
    col1.write(X_data.shape)
    col2.write(Y_data.shape)

    TestingRecords=5
    X_train=X_data[:-TestingRecords]
    X_test=X_data[-TestingRecords:]
    Y_train=Y_data[:-TestingRecords]
    Y_test=Y_data[-TestingRecords:]

    st.header('Training and Testing Data Shapes')
    col1,col2=st.columns(2)
    col1.write(X_train.shape)
    col2.write(Y_train.shape)
    col1.write(X_test.shape)
    col2.write(Y_test.shape)

    





    

