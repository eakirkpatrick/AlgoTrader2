import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from dataCleaner import DataHandler


class LSTMModel:

	def __init__(self, file, epochs, output_space, dropout_perc, batch_size, input_shape):
		self.dataset_train = pd.read_csv(file)
		self.epochs = epochs
		self.output_space = output_space
		self.dropout_percent = dropout_perc
		self.batch_size = batch_size
		self.input_shape = input_shape

	def buildModel(self, number_of_layers):
		model = Sequential()
		model.add(LSTM(units=self.output_space,
			return_sequences=True, input_shape=self.input_shape
			)

		model.add(Dropout(self.dropout_percent))


		for i in range(number_of_layers-1):
			model.add(LSTM(units=self.output_space, return_sequences=True))
			model.add(Dropout(self.dropout_percent))

		model.add(LSTM(units=self.output_space))
		model.add(Dropout(self.dropout_percent))

		#output layer
		model.add(Dense(units=1))

		#create the model
		model.compile(optimizer='adam', loss='mean_squared_error')

		return model

	def train(x_train, y_train, model):
		trained_model = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
		return trained_model

mydata = DataHandler(10)
test, train = myData.splitData(df)
test, train = myData.normalize(test, train)
x_train, y_train = myData.createInputSequence(train, 4)
x_test, y_test = myData.createInputSequence(test, 4)

model = LSTMModel('thisData.csv', 100, 60, 0.2, 1, len(mydata.train_columns))
myModel = model.buildModel(3)
trained_model = model.train(x_train, y_train, myModel)
