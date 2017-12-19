import pandas as p
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

class ml:

    def regressionTestOne(self):
        # Getting ticker from Quandl
        data = quandl.get('WIKI/GOOGL')

        # Specifying columns to be displayed
        data = data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

        # Setting extra columns HL_PCT and PCT_Change to display the output of specific equation shown below
        data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100.0
        data['PCT_Change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100.0

        # Adding columns into dataframe
        data = data[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

        print(data.head())

    def regressionTestTwo(self):
        # Getting ticker from Quandl
        data = quandl.get('WIKI/GOOGL')

        # Specifying columns to be displayed
        data = data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

        # Setting extra columns HL_PCT and PCT_Change to display the output of specific equation shown below
        data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100.0
        data['PCT_Change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100.0

        # Adding columns into dataframe
        data = data[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

        forecast_col = 'Adj. Close'
        data.fillna(-99999, inplace=True)

        forecast_out = int(math.ceil(0.01*len(data)))

        data['label'] = data[forecast_col].shift(-forecast_out)
        data.dropna(inplace=True)
        print(data.tail())

    def regressionTestThree(self):
        # Getting ticker from Quandl
        data = quandl.get('WIKI/GOOGL')

        # Specifying columns to be displayed
        data = data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

        # Setting extra columns HL_PCT and PCT_Change to display the output of specific equation shown below
        data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100.0
        data['PCT_Change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100.0

        # Adding columns into dataframe
        data = data[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

        forecast_col = 'Adj. Close'
        data.fillna(-99999, inplace=True)

        forecast_out = int(math.ceil(0.01 * len(data)))
        print("Forecasted output: ",forecast_out)

        data['label'] = data[forecast_col].shift(-forecast_out)
        data.dropna(inplace=True)

        x = np.array(data.drop(['label'], 1))
        y = np.array(data['label'])
        x = preprocessing.scale(x)
        y = np.array(data['label'])

        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
        classifier = LinearRegression()
        classifier.fit(x_train, y_train)
        accuracy = classifier.score(x_test, y_test)

        print("Predicted accuracy: ",accuracy)

    def main(self):
        #self.regressionOne()
        #print("\n")
        #self.regressionTwo()
        #print("\n")
        self.regressionThree()



if __name__ == '__main__':
    mlreg = ml()
    mlreg.main()

