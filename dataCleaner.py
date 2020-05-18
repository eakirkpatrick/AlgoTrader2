from binanceApi import APIWrapper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataHandler:

    def __init__(self, time_steps):
        self.train_columns = ['OpenPrice', 'High', 'Low', 'Close', 'Volume', 'QuoteAssetVolume', 'Number of Trades']
        self.time_steps = time_steps #time steps the algo will look back

	self.sc = MinMaxScaler(feature_range=(0,1))

    def handle(self, data):
        prices = np.array(data)
        df = pd.DataFrame({'OpenTime': prices[:,0].astype('float32'), 'OpenPrice': prices[:,1].astype('float32'), 'High': prices[:,2].astype('float32'), 'Low':
        prices[:,3].astype('float32'), 'Close': prices[:,4].astype('float32'), 'Volume': prices[:,5].astype('float32'), 'CloseTime': prices[:,6].astype('float32'), 'QuoteAssetVolume': prices[:,7].astype('float32'),
        'Number of Trades': prices[:,8].astype('float32')})
        #print(df)
        return df

    def normalize(self, df_test, df_train):
        x = df_train.loc[:, self.train_columns].values
        x_train = self.sc.fit_transform(x)
        x_test = self.sc.transform(df_test.loc[:,self.train_columns])

        return x_train, x_test

    def graph(self, df):
        plt.plot(df['OpenPrice'].astype(float))
        plt.show()

    def label(self, df):
        df['Action'] = None
        for i in range(1,len(df)-1):
            if df.loc[i-1, 'OpenPrice'] < df.loc[i, 'OpenPrice'] and df.loc[i, 'OpenPrice'] > df.loc[i+1, 'OpenPrice']:
                df.loc[i, 'Action'] = 'sell'
            elif df.loc[i-1, 'OpenPrice'] > df.loc[i, 'OpenPrice'] and df.loc[i, 'OpenPrice'] < df.loc[i+1, 'OpenPrice']:
                df.loc[i, 'Action'] = 'buy'
            else:
                df.loc[i, 'Action'] = 'hold'

        return df

    def createInputSequence(self, matrix, y_index):
        #matrix is 2d numpy array
        dim_0 = matrix.shape[0] - self.time_steps
        #dim_1 is number of input variables
        dim_1 = matrix.shape[1]

        input = np.zeros((dim_0, self.time_steps, dim_1))
        output = np.zeros((dim_0,))

        for i in range(dim_0):
            input[i] = matrix[i:self.time_steps+i]
            output[i] = matrix[self.time_steps+i, y_index]

        return input, output

    def padList(self, input):
        raggedTensor = tf.ragged.constant(input)
        return raggedTensor

    def createLabel(self, df):
        labels = []
        for i in range(2, len(df)):
            labels.append(df.loc[i, 'Action'])
        return labels

    def splitData(self, df):
        df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)
        return df_train, df_test

# test = APIWrapper('BTCBUSD')
#
# content = test.collectData(1825)
# df = pd.read_csv('thisData.csv')
#
# myData = DataHandler(df, 10)
# # df.to_csv('thisData.csv')
# test, train = myData.splitData(df)
# test, train = myData.normalize(test, train)
# x_train, y_train = myData.createInputSequence(train, 4)
# x_test, y_test = myData.createInputSequence(test, 4)



#myData.graph(df)
