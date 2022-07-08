import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

nInst = 100
currentPos = np.zeros(nInst)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


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

    dicDates = pd.DataFrame(dates)      #this section is making the dataframe of 1 stock
    d = dict(enumerate(prcAll[0].flatten(), 1))
    df = pd.DataFrame(d, index=['ds','y'])
    df = df.transpose()
    df['ds'] = dicDates                 #adding the dates column in for .fit() requirements


    df_train = df
    df_train = df_train.rename(columns={"Date": "y", "Close": "ds"})

    print(df_train)

    m = Prophet()
    m.fit(df_train)                     #now i have to compile results
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    print(forecast)
    a = forecast['yhat']
    b = prcAll[0]
    plt.plot(a)
    plt.plot(b)
    plt.grid()
    plt.show()

    return currentPos

# graphs
# a = prcAll[20]
# plt.plot(a)
# plt.grid()
# plt.show()

# df = pd.DataFrame(pChange)
# print(df)
