# Library imports
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Table
startDate = '2014-01-01'
endDate = '2019-12-31'

# Dow30 stocks in 2018
nameList = [
    'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'XOM',
    'GS', 'HD', 'INTC', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE',
    'PFE', 'PG', 'TRV', 'UNH', 'RTX', 'VZ', 'V', 'WBA', 'WMT', 'DIS'
    ]


data = yf.download(nameList, startDate, endDate, group_by='tickers')
stockSelectionData = yf.download(nameList, '2018-01-01', '2018-12-31', group_by='tickers')

class Stock:
    def __init__(self, name, closeDifferential, highLowDifferential, volume):
        self.name = name
        self.closeDifferential = closeDifferential
        self.highLowDifferential = highLowDifferential
        self.volume = volume

stocksArray = []

for name in nameList:
    stockLow = 0
    stockHigh = 0
    stockTotalVolume = 0

    for value in stockSelectionData[name]['Open']:
        if value > stockHigh or stockHigh == 0:
            stockHigh = value

    for value in stockSelectionData[name]['Close']:
        if value < stockLow or stockLow == 0:
            stockLow = value

    for value in stockSelectionData[name]['Volume']:
        stockTotalVolume += value

    stocksArray.append(Stock(name, stockSelectionData[name]['Close']['2018-12-28'] - stockSelectionData[name]['Close']['2018-01-02'], stockHigh - stockLow, stockTotalVolume))

stocksPositiveDifferential = sorted(stocksArray, key=lambda x: x.closeDifferential, reverse=True)[:3]
stocksNegativeDifferential = sorted(stocksArray, key=lambda x: x.closeDifferential)[:3]
stocksMostVolatile = sorted(stocksArray, key=lambda x: x.highLowDifferential, reverse=True)[:3]
stocksLeastVolatile = sorted(stocksArray, key=lambda x: x.highLowDifferential)[:3]
stocksMostVolume = sorted(stocksArray, key=lambda x: x.volume, reverse=True)[:3]

print('Top 3 most positive stocks ' + str(stocksPositiveDifferential[0].name) + ': ' + str(stocksPositiveDifferential[0].closeDifferential) + ', ' + str(stocksPositiveDifferential[1].name) + ': ' + str(stocksPositiveDifferential[1].closeDifferential) + ', ' + str(stocksPositiveDifferential[2].name) + ': ' + str(stocksPositiveDifferential[2].closeDifferential))
print('Top 3 most negative stocks ' + str(stocksNegativeDifferential[0].name) + ': ' + str(stocksNegativeDifferential[0].closeDifferential) + ', ' + str(stocksNegativeDifferential[1].name) + ': ' + str(stocksNegativeDifferential[1].closeDifferential) + ', ' + str(stocksNegativeDifferential[2].name) + ': ' + str(stocksNegativeDifferential[2].closeDifferential))
print('Top 3 most volatile stocks ' + str(stocksMostVolatile[0].name) + ': ' + str(stocksMostVolatile[0].highLowDifferential) + ', ' + str(stocksMostVolatile[1].name) + ': ' + str(stocksMostVolatile[1].highLowDifferential) + ', ' + str(stocksMostVolatile[2].name) + ': ' + str(stocksMostVolatile[2].highLowDifferential))
print('Top 3 least volatile stocks ' + str(stocksLeastVolatile[0].name) + ': ' + str(stocksLeastVolatile[0].highLowDifferential) + ', ' + str(stocksLeastVolatile[1].name) + ': ' + str(stocksLeastVolatile[1].highLowDifferential) + ', ' + str(stocksLeastVolatile[2].name) + ': ' + str(stocksLeastVolatile[2].highLowDifferential))
print('Top 3 most traded stocks ' + str(stocksMostVolume[0].name) + ': ' + str(stocksMostVolume[0].volume) + ', ' + str(stocksMostVolume[1].name) + ': ' + str(stocksMostVolume[1].volume) + ', ' + str(stocksMostVolume[2].name) + ': ' + str(stocksMostVolume[2].volume))

stockName = stocksNegativeDifferential[2].name
stockData = data[stockName]

stockData

# Graph
plt.figure(figsize=(20, 10))
plt.title(stockName + ' Price History')
plt.plot(stockData['Close'])
plt.xlabel('Year')
plt.ylabel('Price $')

# Training Set
trainingData = math.ceil(len(stockData['Close'].values)* (5/6))

scaler = StandardScaler()
scaledData = scaler.fit_transform(stockData['Close'].values.reshape(-1,1))
trainData = scaledData[0: trainingData, :]

xTrain = []
yTrain = []

for i in range(60, len(trainData)):
    xTrain.append(trainData[i-60:i, 0])
    yTrain.append(trainData[i, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))


# Test Set
testData = scaledData[trainingData-60: , : ]
xTest = []
yTest = stockData['Close'].values[trainingData:]

for i in range(60, len(testData)):
  xTest.append(testData[i-60:i, 0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

# LSTM Model
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

# LSTM Train Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, batch_size= 1, epochs=3)

# Model Evaluation
predictions = model.predict(xTest)
predictions = scaler.inverse_transform(predictions)
print('Model Evaluation for:', stockName)
print('MAE(%)', (metrics.mean_absolute_error(yTest, predictions)/yTest.mean())*100)
print('MSE', metrics.mean_squared_error(yTest, predictions))
print('RMSE', np.sqrt(metrics.mean_squared_error(yTest, predictions)))
print('R2', metrics.r2_score(yTest, predictions))



# Line Chart
data = stockData.filter(['Close'])
train = data[:trainingData]
validation = data[trainingData:]
validation['Predictions'] = predictions
plt.figure(figsize=(20,10))
plt.title('Stock Prediction Model for ' + stockName)
plt.xlabel('Year')
plt.ylabel('Close Price ($)')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()



# Trading Algorithm
PercentChange = 0
# 1258 is the first day of the 2019 year value wise
rightDays = 0
wrongDays = 0

PupAup = 0
PupAdown = 0
PdownAup = 0
PdownAdown = 0

for i in range(0, predictions.size-1):
    currentDay = stockData['Close'].values[1258 + i]
    nextDay = stockData['Close'].values[1258 + i + 1]

    if(predictions[i] > predictions[i+1]): # Next day is predicted to be lower than today
        PercentChange -= (((currentDay - nextDay) / nextDay))
    elif(predictions[i] < predictions[i+1]): # Next day is predicted to be higher than today
        PercentChange += (((nextDay - currentDay) / currentDay))

    if((predictions[i] > predictions[i+1] and currentDay > nextDay) or (predictions[i] < predictions[i+1] and currentDay < nextDay)):
        rightDays += 1
    else:
        wrongDays += 1

    if((predictions[i] < predictions[i+1] and currentDay < nextDay)):
        PupAup += 1
    if((predictions[i] > predictions[i+1] and currentDay > nextDay)):
        PdownAdown += 1
    if((predictions[i] < predictions[i+1] and currentDay > nextDay)):
        PupAdown += 1
    if((predictions[i] > predictions[i+1] and currentDay < nextDay)):
        PdownAup += 1

print('Using Trading Algorithm: ', PercentChange * 100, "%")
print('Holding: ', (((stockData['Close'].values[-1] - stockData['Close'].values[1258]) / stockData['Close'].values[1258]) * 100), "%")
print('Days predicted right: ' + str(rightDays))
print('Days predicted wrong: ' + str(wrongDays))
print('Days predicted up and actual went up: ' + str(PupAup))
print('Days predicted down and actual went down: ' + str(PdownAdown))
print('Days predicted up and actual went down: ' + str(PupAdown))
print('Days predicted down and actual went up: ' + str(PdownAup))
# print('UpUp: ' + str(PupAup) + ' DownDown: ' + str(PdownAdown) + ' UpDown: ' + str(PupAdown) + ' DownUp: ' + str(PdownAup))