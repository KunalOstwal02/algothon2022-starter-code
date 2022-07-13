import numpy as np
import pandas as pd
import time
from datetime import date, timedelta
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

nInst = 100
currentPos = np.zeros(nInst)
tStart = time.time()



def getMyPosition(prcSoFar):
    global currentPos

    def loadPrices(fn):  # grabs prices from prices.txt
        global nt, nInst
        df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
        nt, nInst = df.values.shape
        return (df.values).T

    pricesFile = "./prices.txt"
    prcAll = loadPrices(pricesFile)  # data stored in prcAll

    #-------------------------------------------------------------for the dates
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)
    dates = []
    start_date = date(2015, 1, 1)
    end_date = date(2015, 9, 9)
    for single_date in daterange(start_date, end_date):
        dates.append(single_date.strftime("%Y-%m-%d"))
    #--------------------------------------------------------------

    #                   THE COMMENTED SECTION BELOW RUNS EVERY STOCK PREDICTION, TAKES TOO LONG TO RUN IT REPEATEDLY
    # def predict(i):
    #     dicDates = pd.DataFrame(dates)      #this section is making the dataframe of 1 stock
    #     d = dict(enumerate(prcAll[i].flatten(), 1))
    #     df = pd.DataFrame(d, index=['ds','y'])
    #     df = df.transpose()
    #     df['ds'] = dicDates                 #adding the dates column in for .fit() requirements
    #
    #
    #     df_train = df
    #     df_train = df_train.rename(columns={"Date": "y", "Close": "ds"})
    #
    #     m = Prophet()
    #     m.fit(df_train)                     #now i have to compile results
    #     future = m.make_future_dataframe(periods=1)
    #     forecast = m.predict(future)
    #     #pd.set_option("display.max_rows", None, "display.max_columns", None) #displays whole table in console
    #     return forecast['yhat']
    #
    # predictions = []
    # for i in range(0,len(prcAll)):
    #     predictions.append(predict(i))

          # CODE TO RUN ONE PREDICTION
    dicDates = pd.DataFrame(dates)  # this section is making the dataframe of 1 stock
    d = dict(enumerate(prcAll[0].flatten(), 1))         #change the indicie in prcAll[] for every stock indicie
    df = pd.DataFrame(d, index=['ds','y'])
    df = df.transpose()
    df['ds'] = dicDates                 #adding the dates column in for .fit() requirements


    df_train = df
    df_train = df_train.rename(columns={"Date": "y", "Close": "ds"})

    m = Prophet()
    m.fit(df_train)                     #now i have to compile results
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)    #the prediction, im looking at 'yhat'
    #pd.set_option("display.max_rows", None, "display.max_columns", None) #displays whole table in console



    a = forecast['yhat']    #prediction
    b = pd.DataFrame(prcAll[0])           #stock indices


    #find gradient of last 5 days.
    temp = []
    for i in range(0,len(prcAll),5):
        temp.append(a[i])

    g = np.gradient(temp)
    #if gradient[i] and gradient [i-1]>0, buy, and vice versa

    #.iloc[row][column?]
    totalCashSpent = 0
    totalTrades = 0
    for i in range(0,len(g)):
        if i>0:
            if g[i] > 0 and g[i-1]>0:
                print('buy ')
                currentPos[0] += 500/b.iloc[i*5][0]
                totalCashSpent += 500
                totalTrades += 1
            if g[i] < 0 and g[i-1]<0:
                print('sell ')
                currentPos[1] -= 500 / b.iloc[i*5][0]
                totalCashSpent += 500
                totalTrades += 1


    revenue = currentPos[0] * b.iloc[-1][0]
    print("Position: $"+str(revenue))
    print('Minus original stake: $'+str(totalCashSpent))
    print('Profit/Loss: $'+str(revenue-totalCashSpent))
    print('P/L %: '+str((100*(revenue-totalCashSpent)/totalCashSpent)))
    print('Total Trades: '+str(totalTrades))





    # b = prcAll[0]           #stock indicie
    # plt.plot(a, label = 'Forecasted')
    # plt.plot(b, label = 'Actual')
    # plt.legend()
    # plt.title("Stock Price")
    # plt.xlabel("Days")
    # plt.ylabel("Price")
    # plt.grid()
    # plt.show()


    tEnd = time.time()
    tRun = tEnd - tStart
    print("runTime  : %.3lf " % tRun)

    return currentPos

# graphs
# a = prcAll[20]
# plt.plot(a)
# plt.grid()
# plt.show()

# df = pd.DataFrame(pChange)
# print(df)
