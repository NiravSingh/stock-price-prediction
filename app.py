import streamlit as st
import pandas_datareader as pdr
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

def download_data_csv(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="5y")
    data.reset_index(inplace=True)
    csv_data = data.to_csv(index=False)
    return csv_data

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

import io

def gr1(df0, trainPredictPlot, testPredictPlot):
    plt.plot(df0)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    
    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Display the plot using st.image()
    st.image(buffer)
    
    return plt

def gr2(df3):
    plt.plot(df3)
    
    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Display the plot using st.image()
    st.image(buffer)
    
    return plt


def predictor(name="TSLA",time=100,duration=30):
    file= download_data_csv(name)
    df=pd.read_csv(StringIO(file))
    df1 = df.reset_index()['Close']
    op1 = df1;
    op10 = df1.iloc[-1] ;
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    
    training_size=int(len(df1)*0.75)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    
    time_step = time
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=64,verbose=1)
    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    
    
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    
    
    look_back=time_step
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    
    # plot baseline and predictions
    df0=scaler.inverse_transform(df1)
    g1=gr1(df0,trainPredictPlot,testPredictPlot)
    
    
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    
    lst_output=[]
    n_steps=time_step
    i=0
    while(i<duration):

        if(len(temp_input)>n_steps):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
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
    day_new=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+duration)   
    
    df0 = scaler.inverse_transform(df1[len(df1)-time_step:])
    df9 = scaler.inverse_transform(lst_output)
    #g2=gr2(day_new,df0,day_pred,df9)
    #op5=plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-time_step:]))
    #op6=plt.plot(day_pred,scaler.inverse_transform(lst_output))
    
    df3=df1.tolist()
    df3.extend(lst_output)
    #op7=plt.plot(df3[1200:])
    df3=scaler.inverse_transform(df3).tolist()
    op8=gr2(df3)
    
    lst_output=scaler.inverse_transform(lst_output)
    op9=lst_output[ len(lst_output) - 1]
    return g1,op8,op9,op10    

def main():
    # Set the title and sidebar options
    st.title('Stock Predictor App')
    st.sidebar.title('Input Parameters')

    # Define the input fields in the sidebar
    name = st.sidebar.text_input('Stock Name', 'TSLA')
    time = st.sidebar.number_input('Time Steps', min_value=100, value=300)
    duration = st.sidebar.number_input('Duration', min_value=1, value=50)

    # Call the predictor function when the 'Predict' button is clicked
    if st.sidebar.button('Predict'):
        g1, op8, op9 ,op10 = predictor(name, time, duration)

        # Display the first graph (g1)
        #st.write('Graph 1:')
        #st.pyplot(g1)

        # Display the second graph (op8)
        #st.write('Graph 2:')
        #st.pyplot(op8)
        st.write('Last Closing Value:')
        st.write(op10)
        # Display the predicted value (op9)
        st.write('Predicted Value:')
        st.write(op9)

if __name__ == '__main__':
    main()
