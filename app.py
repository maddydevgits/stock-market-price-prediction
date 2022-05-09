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

    

