# import libraries
from ast import Try
import math
from unittest import expectedFailure
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt
from tickerAttributes import info

model = load_model('keras_modelv2.h5')

st.title('Stock Price Prediction')
START_DATE = '2008-01-01' 
END_DATE = str(dt.datetime.now().strftime('%Y-%m-%d'))

# Get the stock quote
ticker_name = st.text_input('Enter Stock Ticker')
exchange_selection = st.selectbox(
  'Select Stock Exchange',
  ('NSE', 'BSE', 'Other')

)
stock_abbr = {'NSE':'.NS', 'BSE':'.BO', 'Other':''}
ticker_name = ticker_name + stock_abbr[exchange_selection]

ticker = yf.Ticker(ticker_name)
# info = ticker.info
info = info(ticker_name)

df = ticker.history(start = START_DATE, end = END_DATE)

def ticker_attributes(info):
  companyName = info['longName']
  timeZone = info['timeZoneFullName']
#   website = info['website']
  current_price = info['currentPrice']
  previous_close = info['previousClose']
  financial_currency = info['currency']
  dayHigh = info['dayHigh']

  st.write(f"Name of Company: {companyName}")
  st.write(f'Time Zone: {timeZone}')
#   st.write(f'Official Website: {website}')
  st.write(f'Day high: {dayHigh}')
  st.write(f'Previous close: {previous_close} {financial_currency}')
  st.write(f'current Price: {current_price} {financial_currency}')

def close_time_graph(df):
  st.subheader('Closing Price vs Time')
  fig = plt.figure(figsize = (12,6))
  plt.plot(df.Close, 'b')
  st.pyplot(fig)

def ma100_ma200(df):
  st.subheader('Closing Price vs Time chart with 100MA and 200MA')
  ma100 = df.Close.rolling(100).mean()
  ma200 = df.Close.rolling(200).mean()
  fig = plt.figure(figsize = (12,6))
  plt.plot(ma100, 'r', label='100 MOVING AVERAGE')
  plt.plot(ma200, 'g', label = '200 MOVING AVERAGE')
  plt.plot(df.Close, 'b', label = 'CLOSING PRICE')
  plt.legend()
  st.pyplot(fig)

def pred_original_graph(train, valid):
  st.subheader('Prediction Price with compare to Original Price')
  fig2 = plt.figure(figsize = (16,8))
  plt.title('Model')
  plt.xlabel('Date', fontsize = 18)
  plt.ylabel('Close Price in INR', fontsize = 18)
  plt.plot(train['Close'])
  plt.plot(valid[['Close', 'Predictions']])
  plt.legend(['Train', 'Val', 'Predictions'])
  st.pyplot(fig2)


def make_prediction(x_test):
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)
  return predictions

def get_next_prediction(data, model):
  last_60_days = data[-60:].values
  last_60_days_scaled = scaler.transform(last_60_days)
  X_test = []
  X_test.append(last_60_days_scaled)

  #convert to numpy array
  X_test = np.array(X_test)
  #Reshape the data
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  pred_price = model.predict(X_test)
  pred_price = scaler.inverse_transform(pred_price)

  return pred_price

def show_prediction(ticker_name, pred_price, info):
  # ticker_name = ticker_name.split(".")[0]
  ticker_name = info['longName']
  currency = info['currency']
  pred_price = pred_price[0][0]
  next_date = dt.date.today() + dt.timedelta(days = 1)
  st.subheader(f'Predicted Price for {ticker_name}:')
  st.write(f"price prediction for {ticker_name} on {next_date} is {pred_price:.2f} {currency}")


try:
  ticker_attributes(info)
  close_time_graph(df)
  ma100_ma200(df)
  

  # Create new data frame with only Close column
  data = df.filter(['Close'])
  dataset = data.values
  #divide 80% data into training data
  training_data_len = math.ceil(len(dataset)*  .8)
  scaler = MinMaxScaler(feature_range = (0,1))
  scaled_data = scaler.fit_transform(dataset)

  # Create testing data set
  test_data = scaled_data[training_data_len - 60: :]
  #create the data sets x_test and y_test
  x_test = []
  y_test = dataset[training_data_len: :]
  for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

  #convert the array to numpy array 
  x_test = np.array(x_test)

  # Reshape the data
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  train = data[:training_data_len]
  valid = data[training_data_len:]
  predictions = make_prediction(x_test)
  valid['Predictions'] = predictions

  pred_price = get_next_prediction(data, model)
  pred_original_graph(train, valid)
  show_prediction(ticker_name, pred_price, info)
except KeyError:
  st.write("Ticker not found")