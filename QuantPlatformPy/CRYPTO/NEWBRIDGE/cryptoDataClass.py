# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:29:22 2020
"""

import importlib, importlib.machinery
import sys
#import setuptools, imp, csv
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
from numpy import inf
import datetime as dt  
from datetime import datetime#, timtedelta
import matplotlib.pyplot as plt
#import seaborn as sns
import requests

#______________________________________________________________________________        
class dataStructureCryptoCompare():

    def __init__(self,  ccy1=None, ccy2=None, nbOfObs=None, timeframe=None, \
                 exchange=None, data=None, datapoint=None, datadf=None, dfO=None,\
                 dfH=None,dfL=None, dfC=None, dfVfrom=None, dfVto=None,\
                 histDataDir=None):
        """
        .........
        """
        # Address of historical data
        self.histDataDir = histDataDir
        # Tickers
        self.ccy1 = ccy1
        self.ccy2 = ccy2
        # Length of the time serries
        self.nbOfObs = nbOfObs
        # Frequency (daily, hourly, ...)
        self.timeframe = timeframe             
        
        # Point observation for one cross
        self.exchange = exchange
        self.datapoint = datapoint        
        # Time series for one cross
        self.data = data
        self.datadf = datadf
        # DataFrame Open,High,Low,Close for a set of cryptos
        self.dfO = dfO
        self.dfH = dfH
        self.dfL = dfL
        self.dfC = dfC
        self.dfVfrom = dfVfrom
        self.dfVto = dfVto
        #self.o = o
        #self.h = h
        #self.l = l
        #self.c = c
        #self.vfrom = vfrom
        #self.vto = vto
    #__________________________________________________________________________
    # GET CURRENT PRICE DATA
    def get_current_data_OneCrypto(self, exchange=None):
        # API URL
        baseurl = 'https://min-api.cryptocompare.com/data/price'    
        # Tickers
        from_sym = self.ccy1
        to_sym = self.ccy2
        from_sym = 'BTC'
        to_sym = 'USD'
        parameters = {'fsym': from_sym,'tsyms': to_sym}
        # Exchange
        if exchange:
            #print('exchange: ', self.exchange)
            parameters['e'] = self.exchange
        # Response comes as json
        response = requests.get(baseurl, params=parameters)   
        dtemp = response.json()
        if isinstance(from_sym,str) and isinstance(to_sym,str):
            datapoint = {from_sym+to_sym:dtemp[to_sym]}
        else:
            mystring = ''.join(from_sym+to_sym)
            ccy2str = ''.join(to_sym)
            datapoint = {mystring:dtemp[ccy2str]}        
        # Retrieve
        self.datapoint= datapoint#lastDataPoint     
        
    #__________________________________________________________________________
    # HISTORICAL DATA
    def get_hist_data(self, aggregation=1, exchange=None):
        # Time frame
        timeframe = self.timeframe
        # API URL
        if timeframe == 'd' or timeframe == 'day' or timeframe == 'daily':
            baseurl = 'https://min-api.cryptocompare.com/data/v2/histo'
        elif timeframe == 'h' or timeframe == 'hour' or timeframe == 'hourly':     
            #baseurl = 'https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=10
            baseurl = 'https://min-api.cryptocompare.com/data/v2/histohour'
        baseurl += timeframe
        # Tickers
        from_sym = self.ccy1
        to_sym = self.ccy2        
        #Parameters
        limit = self.nbOfObs
        parameters = {'fsym': from_sym,
                      'tsym': to_sym,
                      'limit': limit,
                      'aggregate': aggregation}
        if exchange:
            #print('exchange: ', exchange)
            parameters['e'] = exchange    
        
        print('baseurl: ', baseurl) 
        print('timeframe: ', timeframe)
        print('parameters: ', parameters)
        
        # response comes as json
        response = requests.get(baseurl, params=parameters)   
        
        data = response.json()['Data']['Data'] 
        
        self.data = data      
  
    #__________________________________________________________________________
    # CONVERT TO DATAFRAME
    def data_to_dataframe(self):
        #Retrieve data
        data = self.data
        #data from json is in array of dictionaries
        df = pd.DataFrame.from_dict(data)
        
        # time is stored as an epoch, we need normal dates
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        print(df.tail())
        
        self.df = df
      
    #______________________________________________________________________________
    def plot_data(self):
        # got his warning because combining matplotlib 
        # and time in pandas converted from epoch to normal date
        # To register the converters:
        # 	>>> from pandas.plotting import register_matplotlib_converters
        # 	>>> register_matplotlib_converters()
        #  warnings.warn(msg, FutureWarning)
        
        df = self.df
        ccy1 = self.ccy1
        ccy2 = self.ccy2
        
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        
        if isinstance(ccy1,str):
            ccy1=ccy1
            ccy2 = ccy2
        else:
            ccy1 = ''.join(ccy1) 
            ccy2 = ''.join(ccy2)

        plt.figure(figsize=(15,5))
        plt.title('{} / {} price data'.format(ccy1, ccy2))
        plt.plot(df.index, df.close)
        plt.legend()
        plt.show()
        
        return None
    
    #__________________________________________________________________________
    # HISTORICAL SINGLE CRYPTO DATA FOR SET
    def get_histData_SingleCrypto_for_Set(self, baseCcy, quoteCcy):
        # -- Step 1: download data --------------------------------------------
        # Time frame
        timeframe = self.timeframe
        # API URL
        baseurl = 'https://min-api.cryptocompare.com/data/v2/histo'
        baseurl += timeframe    
        #Parameters
        aggregation = 1
        limit = self.nbOfObs
        #limit = mynbOfObs
        parameters = {'fsym': baseCcy,
                      'tsym': quoteCcy,
                      'limit': limit,
                      'aggregate': aggregation}
        exchange=None
        if exchange:
            parameters['e'] = exchange    
        # response comes as json
        response = requests.get(baseurl, params=parameters)   
        data = response.json()['Data']['Data'] 
        # -- Step 2: transform to data frame ----------------------------------
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df  
    
    #__________________________________________________________________________
    # HISTORICAL SINGLE CRYPTO DATA FOR SET
    def get_histData_SingleCrypto_for_Set_V2(self, baseurl, baseCcy,
                                    quoteCcy, aggregation, limit):
        # -- Step 1: download data -- 
        #Parameters
        parameters = {'fsym': baseCcy,
                      'tsym': quoteCcy,
                      'limit': limit,
                      'aggregate': aggregation}
        # response comes as json
        response = requests.get(baseurl, params=parameters)   
        data = response.json()['Data']['Data'] 
        # -- Step 2: transform to data frame --
        df = pd.DataFrame.from_dict(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # -- Retrieve --
        return df      
    
    #__________________________________________________________________________
    # HISTORICAL SINGLE CRYPTO DATA FOR LOOP
    def hourly_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
        url = 'https://min-api.cryptocompare.com/data/v2/histohour?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        #url = 'https://min-api.cryptocompare.com/data/v2/histohour?fsym={}&tsym={}&limit={}'\
        #    .format(symbol.upper(), comparison_symbol.upper(), limit)    
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        return df    
    # - Usage -
    #time_delta = 1;
    #df = hourly_price_historical('BTC', 'USD', 9999, time_delta)
    #print('Max length = %s' % len(df))
    #print('Max time = %s' % (df.timestamp.max() - df.timestamp.min()))
    #plt.plot(df.timestamp, df.close)
    #plt.xticks(rotation=45)
    #plt.show()
    
    #__________________________________________________________________________
    # INTERSECT / ALLIGN DATA  
    def joinDataframes_Wrapper(self, lhs_Df, rhs_Df, colBenchName, colNameTarget, 
                              colTargetNewName, joinMethod, convertDate):
        # -- Extract target column --------------------------------------------
        rhs_Df2 = pd.DataFrame({colBenchName: rhs_Df[colBenchName],
                                colNameTarget:rhs_Df[colNameTarget]})
        # -- Rename new column ------------------------------------------------
        rhs_Df2 = rhs_Df2.rename(columns={colNameTarget:colTargetNewName})
        # -- Date conversion if needed ----------------------------------------
        if convertDate == 'object2datetime64':
            rhs_Df2[colBenchName] = pd.to_datetime(rhs_Df2[colBenchName])
        # -- Select join method -----------------------------------------------
        if joinMethod == "innerJoins" or joinMethod == "InnerJoins":
            lhs_Df = pd.merge(lhs_Df, rhs_Df2, on='timestamp')
        elif joinMethod == "leftJoins" or joinMethod == "LeftJoins":
            # Outter-left join
            lhs_Df = pd.merge(left=lhs_Df, right=rhs_Df2, how='left',
                              left_on=colBenchName, right_on=colBenchName,
                              validate="1:1")
        # -- Clean data -------------------------------------------------------
        new_Df = lhs_Df.copy()
        # If first row i NaN, replace with 0
        new_Df.iloc[0:1,:] = new_Df.iloc[0:1,:].fillna(0)
        # Replace all NaNs by previous value
        new_Df = new_Df.fillna(method='ffill')
        # -- Return data  -----------------------------------------------------
        return new_Df    

    #__________________________________________________________________________
    # HISTORICAL DATA FOR MANY CRYPTOS
    def downloadCCdata(self):#, aggregation=1, exchange=None):
        
        # -- Step 1.: Set the parameters --------------------------------------
        timeframe = self.timeframe
        limit     = self.nbOfObs
        # ~~ API URL ~~
        #if timeframe == 'd' or timeframe == 'day' or timeframe == 'daily':
        baseurl = 'https://min-api.cryptocompare.com/data/v2/histo'
        #elif timeframe == 'h' or timeframe == 'hour' or timeframe == 'hourly':    
            #baseurl = 'https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=10
        #baseurl = 'https://min-api.cryptocompare.com/data/v2/histohour'        
        #baseurl = 'https://min-api.cryptocompare.com/data/v2/histo'
        baseurl += timeframe    
        # ~~ Aggregation for Parameters ~~
        aggregation = 1   
        # ~~ Tickers & Dimensions ~~
        ccy1 = self.ccy1
        ccy2 = self.ccy2
        #if isinstance(ccy1, list):
        nbOfInst = len(ccy1) 
        
        # -- Step 2.: Create the Benchmarkfor Allignement ---------------------
        # ~~ By convention, use the first ticker ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        baseCcy  = ccy1[0]
        quoteCcy = ccy2[0]
        baseCcy_Name = ''.join(baseCcy)# Used to rename the columns
        # ~~ Download the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dfseed = self.get_histData_SingleCrypto_for_Set(baseCcy, quoteCcy)
        print(baseCcy)
        # ~~ Build the "seed" for intersection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Note: the outputnformat is:
        #              high       low  ...  conversionType  conversionSymbol
        # time                            ...                                  
        # 2017-03-16   1260.24   1118.96  ...          direct                  
        #.... i reset the index so that:
        #            time      high       low  ...     close  conversionType  conversionSymbol
        # 0    2017-03-16   1260.24   1118.96  ...   1172.88          direct         
        # ^^^ Open ^^^
        dfO = dfseed.drop(columns=["high","low","close","conversionType",
                                   "conversionSymbol","volumefrom","volumeto"])
        dfO = dfO.reset_index()
        dfO = dfO.rename(columns={'open':baseCcy_Name})
        # ^^^ High ^^^
        dfH = dfseed.drop(columns=["low","open","close","conversionType",
                                   "conversionSymbol","volumefrom","volumeto"])
        dfH = dfH.reset_index()
        dfH = dfH.rename(columns={'high':baseCcy_Name})
        # ^^^ Low ^^^
        dfL = dfseed.drop(columns=["high","open","close","conversionType",
                                   "conversionSymbol","volumefrom","volumeto"])
        dfL = dfL.reset_index()
        dfL = dfL.rename(columns={'low':baseCcy_Name})
        # ^^^ Close ^^^
        dfC = dfseed.drop(columns=["high","low","open","conversionType",
                                   "conversionSymbol","volumefrom","volumeto"])
        dfC = dfC.reset_index()  
        dfC = dfC.rename(columns={'close':baseCcy_Name})
        # ^^^ Volume from ^^^
        dfVfrom = dfseed.drop(columns=["high","low","open","close","conversionType",
                                   "conversionSymbol","volumeto"])
        dfVfrom = dfVfrom.reset_index()
        dfVfrom = dfVfrom.rename(columns={'volumefrom':baseCcy_Name})
        # ^^^ Volume To ^^^
        dfVto = dfseed.drop(columns=["high","low","open","close","conversionType",
                                     "conversionSymbol","volumefrom"])
        dfVto = dfVto.reset_index()
        dfVto = dfVto.rename(columns={'volumeto':baseCcy_Name})             
                 
        # -- Step 3.: Download the other crypto-currencies --------------------
        for j in range(1,nbOfInst):
            # Extract the ticker
            baseCcy = ccy1[j]
            quoteCcy = ccy2[j]
            # ~~ Step 3.1.: Download the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            df = self.get_histData_SingleCrypto_for_Set_V2(baseurl, baseCcy,
                                    quoteCcy, aggregation, limit)
            print(baseCcy)
            # ~~ Step 3.2.: intersection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ^^^ Intersect Open ^^^
            dfO = self.joinDataframes_Wrapper(dfO, df,'time','open',baseCcy,
                                               'leftJoins','doNotConvertDate')
            # ^^^ Intersect High ^^^
            dfH = self.joinDataframes_Wrapper(dfH, df,'time','high',baseCcy,
                                               'leftJoins','doNotConvertDate')  
            #  ^^^ Intersect Low ^^^
            dfL = self.joinDataframes_Wrapper(dfL, df, 'time','low',baseCcy,
                                               'leftJoins','doNotConvertDate')        
            #  ^^^ Intersect Close ^^^
            dfC = self.joinDataframes_Wrapper(dfC, df, 'time','close',baseCcy,
                                               'leftJoins','doNotConvertDate')  
            #  ^^^ Intersect Volumefrom ^^^
            dfVfrom = self.joinDataframes_Wrapper(dfVfrom, df, 'time','volumefrom',baseCcy,
                                              'leftJoins','doNotConvertDate') 
            #  ^^^ Intersect Volumeto ^^^
            dfVto = self.joinDataframes_Wrapper(dfVto, df, 'time','volumeto',baseCcy,
                                              'leftJoins','doNotConvertDate')             
            # ~~ Problem with LUNA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # note:   time       close
            #       8/25/2019	0.003752
            #       8/26/2019	1.17700
            # for safety-sake I zero-out all the data before '2019-08-31' 
            # note: 'time' has been added & joining done: ---> LUNA is in j+1
            if baseCcy == 'LUNA':
                try:
                    timeIdx = int(dfC[dfC['time']=='2019-08-31'].index.values)
                    dfO.iloc[0:timeIdx,j+1]   = 0*dfO.iloc[0:timeIdx,j+1]
                    dfH.iloc[0:timeIdx,j+1]   = 0*dfH.iloc[0:timeIdx,j+1]
                    dfL.iloc[0:timeIdx,j+1]   = 0*dfL.iloc[0:timeIdx,j+1]
                    dfC.iloc[0:timeIdx,j+1]   = 0*dfC.iloc[0:timeIdx,j+1]
                    dfVfrom.iloc[0:timeIdx,j+1] = 0*dfVfrom.iloc[0:timeIdx,j+1]
                    dfVto.iloc[0:timeIdx,j+1] = 0*dfVto.iloc[0:timeIdx,j+1]
                except:
                    pass
                        
        # Dimensions
        [nn,nc] = dfC.shape
        # Clean some data
        # Deal with some data problems
        # Clean Close (if jump price higher than maxJump)
        # first check ....
        for i in range(1,nn):
            for j in range(1,nc):
                if dfC.iloc[i-1,j] != 0 and dfC.iloc[i,j] == 0:
                    dfC.iloc[i,j] = dfC.iloc[i-1,j]   
        # ... second check            
        for i in range(1,nn):
            for j in range(1,nc):                    
                if dfC.iloc[i-1,j] != 0 and dfC.iloc[i,j] != 0 and \
                    (dfC.iloc[i,j] > 10 * dfC.iloc[i-1,j] or dfC.iloc[i,j] < 0.1 * dfC.iloc[i-1,j]):
                    dfC.iloc[i,j] = dfC.iloc[i-1,j]
        # Clean Open
        # first check ....
        for i in range(1,nn):
            for j in range(1,nc):
                if dfO.iloc[i-1,j] != 0 and dfO.iloc[i,j] == 0:
                    dfO.iloc[i,j] = dfC.iloc[i-1,j]   
        # ... second check
        for i in range(1,nn):
            for j in range(1,nc):
                if dfO.iloc[i-1,j] != 0 and dfO.iloc[i,j] != 0 and \
                    (dfO.iloc[i,j] > 10 * dfO.iloc[i-1,j] or dfO.iloc[i,j] < 0.1 * dfO.iloc[i-1,j]):
                    dfO.iloc[i,j] = dfC.iloc[i-1,j]                     
        # Transform pandat o numpy array           
        #o = dfO.iloc[:,1:nc].to_numpy()
        #h = dfH.iloc[:,1:nc].to_numpy()
        #l = dfL.iloc[:,1:nc].to_numpy()
        #c = dfC.iloc[:,1:nc].to_numpy() 
        #vfrom = dfVfrom.iloc[:,1:nc].to_numpy() 
        #vto = dfVto.iloc[:,1:nc].to_numpy() 
    
        # -- Step 4.: Build time vector ---------------------------------------        
        def extractTimeVectorFromDataframe(myDf, coltimeName, resetDunResetIndex, convertDate):
            # resetDunResetIndex = ["reset index", "reset"] --> ...
            #    ... does the user want an index or not    
            if resetDunResetIndex=="reset index" or resetDunResetIndex=="reset": 
                myDf = myDf.reset_index()
            #  convertDate = ["datetime64" , "datetime64[D]", "datetime64[ns]" ==> ...
            #    ... convert date 
            # note: for .e.g, for the crypto daily data from CryptoCompare the format is
            # '2017-08-31T00:00:00.000000000' when the reader fetch csv data.
            # to transform this fomrat into '2017-08-31', the user specifies
            # convertDate = "datetime64[D]"
            if convertDate == "datetime64":
                benchDate_np = myDf[coltimeName].to_numpy(dtype='datetime64')
            elif convertDate == "datetime64[ns]":
                benchDate_np = myDf[coltimeName].to_numpy(dtype='datetime64[ns]')        
            elif convertDate == "datetime64[D]":
                benchDate_np = myDf[coltimeName].to_numpy(dtype='datetime64[D]')        
            else:
                benchDate_np = myDf[coltimeName].to_numpy()
            # dataFrame format    
            benchDate_df = myDf[coltimeName]
            return benchDate_df , benchDate_np
        
        benchdate_df, benchdate_np = extractTimeVectorFromDataframe(dfC,"time", "dunresetIndex", "no")
        
        # -- Step 5.: Assign --------------------------------------------------
        self.dfO = dfO
        self.dfH = dfH
        self.dfL = dfL
        self.dfC = dfC
        self.dfVfrom = dfVfrom
        self.dfVto = dfVto
        self.benchdate_df = benchdate_df
        self.benchdate_np = benchdate_np
        #self.o = o
        #self.h = h
        #self.l = l
        #self.c = c
        #self.vfrom = vfrom
        #self.vto = vto            
       
    #__________________________________________________________________________
    # HISTORICAL DATA FOR MANY CRYPTOS
    def downloadCCdata_ConcatHistCSV(self, chosenDirectory):#, aggregation=1, exchange=None):
        
        #histDataDir = chosenDirectory
        
        # -- Step 1.: Set the parameters --------------------------------------
        timeframe = self.timeframe
        limit     = self.nbOfObs
        #histDataDir = self.histDataDir
        # ~~ API URL ~~
        #if timeframe == 'd' or timeframe == 'day' or timeframe == 'daily':
        baseurl = 'https://min-api.cryptocompare.com/data/v2/histo'
        baseurl += timeframe    
        # ~~ Aggregation for Parameters ~~
        aggregation = 1   
        # ~~ Tickers & Dimensions ~~
        ccy1 = self.ccy1
        ccy2 = self.ccy2
        #if isinstance(ccy1, list):
        nbOfInst = len(ccy1) 
        
        # -- Step 2.: Create the Benchmarkfor Allignement ---------------------
        # ~~ By convention, use the first ticker ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        baseCcy  = ccy1[0]
        quoteCcy = ccy2[0]
        baseCcy_Name = ''.join(baseCcy)# Used to rename the columns
        # ~~ Download the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dfseed = self.get_histData_SingleCrypto_for_Set(baseCcy, quoteCcy)
        print(baseCcy)
        # ~~ Build the "seed" for intersection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
        # Open
        dfO = dfseed.drop(columns=["high","low","close","conversionType",
                                   "conversionSymbol","volumefrom","volumeto"])
        dfO = dfO.reset_index()
        dfO = dfO.rename(columns={'open':baseCcy_Name})
        # High
        dfH = dfseed.drop(columns=["low","open","close","conversionType",
                                   "conversionSymbol","volumefrom","volumeto"])
        dfH = dfH.reset_index()
        dfH = dfH.rename(columns={'high':baseCcy_Name})
        # Low
        dfL = dfseed.drop(columns=["high","open","close","conversionType",
                                   "conversionSymbol","volumefrom","volumeto"])
        dfL = dfL.reset_index()
        dfL = dfL.rename(columns={'low':baseCcy_Name})
        # Close
        dfC = dfseed.drop(columns=["high","low","open","conversionType",
                                   "conversionSymbol","volumefrom","volumeto"])
        dfC = dfC.reset_index()  
        dfC = dfC.rename(columns={'close':baseCcy_Name})
        # Volume from
        dfVfrom = dfseed.drop(columns=["high","low","open","close","conversionType",
                                   "conversionSymbol","volumeto"])
        dfVfrom = dfVfrom.reset_index()
        dfVfrom = dfVfrom.rename(columns={'volumefrom':baseCcy_Name})
        # Volume To
        dfVto = dfseed.drop(columns=["high","low","open","close","conversionType",
                                   "conversionSymbol","volumefrom"])
        dfVto = dfVto.reset_index()
        dfVto = dfVto.rename(columns={'volumeto':baseCcy_Name})        
                 
        # -- Step 3.: Download the crypto-currencies --------------------------
        for j in range(1,nbOfInst):
            # Extract the ticker
            baseCcy = ccy1[j]
            quoteCcy = ccy2[j]
            # ~~ Step 3.1.: Download the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            df = self.get_histData_SingleCrypto_for_Set_V2(baseurl, baseCcy,
                                    quoteCcy, aggregation, limit)
            print(baseCcy)
            # ~~ Step 3.2.: intersection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Intersect Open
            dfO = self.joinDataframes_Wrapper(dfO, df,'time','open',baseCcy,
                                                'leftJoins','doNotConvertDate')
            # Intersect High
            dfH = self.joinDataframes_Wrapper(dfH, df,'time','high',baseCcy,
                                                'leftJoins','doNotConvertDate')  
            # Intersect Low
            dfL = self.joinDataframes_Wrapper(dfL, df, 'time','low',baseCcy,
                                                'leftJoins','doNotConvertDate')        
            # Intersect Close
            dfC = self.joinDataframes_Wrapper(dfC, df, 'time','close',baseCcy,
                                                'leftJoins','doNotConvertDate')  
            # Intersect Volumefrom
            dfVfrom = self.joinDataframes_Wrapper(dfVfrom, df, 'time','volumefrom',baseCcy,
                                              'leftJoins','doNotConvertDate') 
            # Intersect Volumeto
            dfVto = self.joinDataframes_Wrapper(dfVto, df, 'time','volumeto',baseCcy,
                                              'leftJoins','doNotConvertDate')             
            # ~~ Problem with LUNA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # note:   time       close
            #       8/25/2019	0.003752
            #       8/26/2019	1.17700
            # for safety-sake I zero-out all the data before '2019-08-31' 
            # note: 'time' has been added & joining done: ---> LUNA is in j+1
            if baseCcy == 'LUNA':
                try:
                    timeIdx = int(dfC[dfC['time']=='2019-08-31'].index.values)
                    dfO.iloc[0:timeIdx,j+1]   = 0*dfO.iloc[0:timeIdx,j+1]
                    dfH.iloc[0:timeIdx,j+1]   = 0*dfH.iloc[0:timeIdx,j+1]
                    dfL.iloc[0:timeIdx,j+1]   = 0*dfL.iloc[0:timeIdx,j+1]
                    dfC.iloc[0:timeIdx,j+1]   = 0*dfC.iloc[0:timeIdx,j+1]
                    dfVfrom.iloc[0:timeIdx,j+1] = 0*dfVfrom.iloc[0:timeIdx,j+1]
                    dfVto.iloc[0:timeIdx,j+1] = 0*dfVto.iloc[0:timeIdx,j+1]
                except:
                    pass
                               
        # -- Step 4: Fetch historical data ------------------------------------
        # Open
        df_histOpen = pd.read_csv(chosenDirectory + "openPrice.csv", header=0, sep=',', parse_dates=['time'])
        #df_histOpen = df_histOpen.set_index(['time'])
        #df_histOpen = df_histOpen.reset_index() --- dfO['time'] = pd.to_datetime(dfO['time'])
        # High
        df_histHigh = pd.read_csv(chosenDirectory + "highPrice.csv", header=0, sep=',', parse_dates=['time'])
        # Low
        df_histLow = pd.read_csv(chosenDirectory + "lowPrice.csv", header=0, sep=',', parse_dates=['time'])      
        # Close
        df_histClose = pd.read_csv(chosenDirectory + "closePrice.csv", header=0, sep=',', parse_dates=['time'])     
        # Volume From
        df_histVfrom = pd.read_csv(chosenDirectory + "volumeFrom.csv", header=0, sep=',', parse_dates=['time'])
        # Volume To
        df_histVto = pd.read_csv(chosenDirectory + "volumeTo.csv", header=0, sep=',', parse_dates=['time'])        
                
        # -- Step 6: Concatenate OHLC & Volume -------------------------------- 
        # Concatenate Open
        lastHistDate = df_histOpen['time'].iloc[-1]
        #tempArray = dfO[dfO['time']==lastHistDate].index.values
        tempList = dfO.index[dfO['time']==lastHistDate].tolist()
        lastHistDate_rowIdxInNew = tempList[0]
        dfO = pd.concat([ df_histOpen.iloc[0:df_histOpen.shape[0]-1, :], dfO.iloc[lastHistDate_rowIdxInNew:, :] ])
        dfO = dfO.set_index(['time'])
        dfO = dfO.reset_index()
        # Concatenate High
        lastHistDate = df_histHigh['time'].iloc[-1]
        tempList = dfH.index[dfH['time']==lastHistDate].tolist()
        lastHistDate_rowIdxInNew = tempList[0]
        dfH = pd.concat([ df_histHigh.iloc[0:df_histHigh.shape[0]-1, :], dfH.iloc[lastHistDate_rowIdxInNew:, :] ]) 
        dfH = dfH.set_index(['time'])
        dfH = dfH.reset_index()       
        # Concatenate Low
        lastHistDate = df_histLow['time'].iloc[-1]
        tempList = dfL.index[dfL['time']==lastHistDate].tolist()
        lastHistDate_rowIdxInNew = tempList[0]
        dfL = pd.concat([ df_histLow.iloc[0:df_histLow.shape[0]-1, :], dfL.iloc[lastHistDate_rowIdxInNew:, :] ])   
        dfL = dfL.set_index(['time'])
        dfL = dfL.reset_index()    
        # Concatenate Close
        lastHistDate = df_histClose['time'].iloc[-1]
        tempList = dfC.index[dfC['time']==lastHistDate].tolist()
        lastHistDate_rowIdxInNew = tempList[0]
        dfC = pd.concat([ df_histClose.iloc[0:df_histClose.shape[0]-1, :], dfC.iloc[lastHistDate_rowIdxInNew:, :] ]) 
        dfC = dfC.set_index(['time'])
        dfC = dfC.reset_index()  
        # Concatenate VolumeFrom
        lastHistDate = df_histVfrom['time'].iloc[-1]
        tempList = dfVfrom.index[dfVfrom['time']==lastHistDate].tolist()
        lastHistDate_rowIdxInNew = tempList[0]
        dfVfrom = pd.concat([ df_histVfrom.iloc[0:df_histVfrom.shape[0]-1, :], dfVfrom.iloc[lastHistDate_rowIdxInNew:, :] ])  
        dfVfrom = dfVfrom.set_index(['time'])
        dfVfrom = dfVfrom.reset_index()  
        # Concatenate VolumeTo
        lastHistDate = df_histVto['time'].iloc[-1]
        tempList = dfVto.index[dfVto['time']==lastHistDate].tolist()
        lastHistDate_rowIdxInNew = tempList[0]
        dfVto = pd.concat([ df_histVto.iloc[0:df_histVto.shape[0]-1, :], dfVto.iloc[lastHistDate_rowIdxInNew:, :] ]) 
        dfVto = dfVto.set_index(['time'])
        dfVto = dfVto.reset_index()  
            
        # -- Step 5: Clean some data --------
        # note: not the best clean for some rare (however rare instances) of 
        #       abnormal data
        # Dimensions
        [nn,nc] = dfC.shape
        # Clean Close (if jump price higher than maxJump)
        # first check ....
        for i in range(1,nn):
            for j in range(1,nc):
                if dfC.iloc[i-1,j] != 0 and dfC.iloc[i,j] == 0:
                    dfC.iloc[i,j] = dfC.iloc[i-1,j]   
        # ... second check            
        for i in range(1,nn):
            for j in range(1,nc):                    
                if dfC.iloc[i-1,j] != 0 and dfC.iloc[i,j] != 0 and \
                    (dfC.iloc[i,j] > 10 * dfC.iloc[i-1,j] or dfC.iloc[i,j] < 0.1 * dfC.iloc[i-1,j]):
                    dfC.iloc[i,j] = dfC.iloc[i-1,j]
        # Clean Open
        # first check ....
        for i in range(1,nn):
            for j in range(1,nc):
                if dfO.iloc[i-1,j] != 0 and dfO.iloc[i,j] == 0:
                    dfO.iloc[i,j] = dfC.iloc[i-1,j]   
        # ... second check
        for i in range(1,nn):
            for j in range(1,nc):
                if dfO.iloc[i-1,j] != 0 and dfO.iloc[i,j] != 0 and \
                    (dfO.iloc[i,j] > 10 * dfO.iloc[i-1,j] or dfO.iloc[i,j] < 0.1 * dfO.iloc[i-1,j]):
                    dfO.iloc[i,j] = dfC.iloc[i-1,j]                     
        # Transform pandat o numpy array           
        #o = dfO.iloc[:,1:nc].to_numpy()
        #h = dfH.iloc[:,1:nc].to_numpy()
        #l = dfL.iloc[:,1:nc].to_numpy()
        #c = dfC.iloc[:,1:nc].to_numpy() 
        #vfrom = dfVfrom.iloc[:,1:nc].to_numpy() 
        #vto = dfVto.iloc[:,1:nc].to_numpy()  

        # -- Step 6.: Time-vector --
        def extractTimeVectorFromDataframeV3(myDf, coltimeName, resetDunResetIndex, convertDate, convertDfDateTo64):
            # resetDunResetIndex = ["reset index", "reset"] --> ...
            #    ... does the user want an index or not    
            if resetDunResetIndex=="reset index" or resetDunResetIndex=="reset": 
                myDf = myDf.reset_index()
            #  convertDate = ["datetime64" , "datetime64[D]", "datetime64[ns]" ==> ...
            #    ... convert date 
            # note: for .e.g, for the crypto daily data from CryptoCompare the format is
            # '2017-08-31T00:00:00.000000000' when the reader fetch csv data.
            # to transform this fomrat into '2017-08-31', the user specifies
            # convertDate = "datetime64[D]"
            if convertDate == 'datetime64' or convertDate == 'datetime64[ns]'\
                or convertDate == 'datetime64[D]':
                benchDate_np = myDf[coltimeName].to_numpy(dtype=convertDate)        
            else:
                benchDate_np = myDf[coltimeName].to_numpy()
            # As above for the dataFrame format
            if convertDfDateTo64 == 'datetime64' or convertDfDateTo64 == 'datetime64[ns]'\
                or convertDfDateTo64 == 'datetime64[D]':
                benchDate_df = myDf[coltimeName].astype(convertDfDateTo64)        
            else:
                benchDate_df = myDf[coltimeName]     
            # Return output
            return benchDate_df , benchDate_np        
        
        benchdate_df, benchdate_np = extractTimeVectorFromDataframeV3(\
                    dfC, "time", "dunresetIndex", "datetime64[D]", 'datetime64[ns]')              

        # -- Step 7: Assign ---------------------------------------------------
        # dataframes
        self.dfO = dfO
        self.dfH = dfH
        self.dfL = dfL
        self.dfC = dfC
        self.dfVfrom = dfVfrom
        self.dfVto = dfVto
        self.benchdate_df = benchdate_df
        self.benchdate_np = benchdate_np          
        # numpyArrays format (because I use them inTalib,...)
        #self.o = o
        #self.h = h
        #self.l = l
        #self.c = c
        #self.vfrom = vfrom
        #self.vto = vto

#______________________________________________________________________________        
class dataStructureCsv():

    # -- INITIALISIATION --
    def __init__(self, sourceDir=None,\
                 dfO=None, dfH=None, dfL=None, dfC=None, \
                 dfVfrom=None, dfVto=None):
                 #o=None,   h=None,   l=None,   c=None, vfrom=None, vto=None,
        # Input
        self.sourceDir = sourceDir
        # OHLCV dataframe
        self.dfO = dfO
        self.dfH = dfH
        self.dfL = dfL
        self.dfC = dfC
        self.dfVfrom = dfVfrom
        self.dfVto = dfVto
        # numpyArray
        #self.o = o
        #self.h = h
        #self.l = l
        #self.c = c
        #self.vfrom = vfrom
        #self.vto = vto
        
    # -- HISTORICAL DATA --
    def getCsvHistoricalData(self, sourceDir):
        
        # -- Step 1.: Extract the sub-universe --
        # Open
        dfO = pd.read_csv(sourceDir + "openPrice.csv", header=0, sep=',', parse_dates=['time'])
        dfO = dfO.set_index(['time'])
        dfO = dfO.reset_index()
        #dfO['time'] = pd.to_datetime(dfO['time'])
        # High
        dfH = pd.read_csv(sourceDir + "highPrice.csv", header=0, sep=',', parse_dates=['time'])
        dfH = dfH.set_index(['time'])
        dfH = dfH.reset_index()
        #dfH['time'] = pd.to_datetime(dfH['time'])
        # Low
        dfL = pd.read_csv(sourceDir + "lowPrice.csv", header=0, sep=',', parse_dates=['time'])
        dfL = dfL.set_index(['time'])
        dfL = dfL.reset_index()      
        #dfL['time'] = pd.to_datetime(dfL['time'])        
        # Close
        dfC = pd.read_csv(sourceDir + "closePrice.csv", header=0, sep=',', parse_dates=['time'])
        dfC = dfC.set_index(['time'])
        dfC = dfC.reset_index() 
        #dfC['time'] = pd.to_datetime(dfC['time'])          
        # Close dfV (Vfrom)
        dfVfrom = pd.read_csv(sourceDir + "volumeFrom.csv", header=0, sep=',', parse_dates=['time'])
        dfVfrom = dfVfrom.set_index(['time'])
        dfVfrom = dfVfrom.reset_index()  
        #dfV['time'] = pd.to_datetime(dfV['time'])
        # Close dfVto
        dfVto = pd.read_csv(sourceDir + "volumeTo.csv", header=0, sep=',', parse_dates=['time'])
        dfVto = dfVto.set_index(['time'])
        dfVto = dfVto.reset_index()     
        #dfVto['time'] = pd.to_datetime(dfVto['time'])
        
        # -- Step 2.: Transform panda to numpy array --
        #[nn,nc] = dfC.shape
        #o = dfO.iloc[:,1:nc].to_numpy()
        #h = dfH.iloc[:,1:nc].to_numpy()
        #l = dfL.iloc[:,1:nc].to_numpy()
        #c = dfC.iloc[:,1:nc].to_numpy() 
        #vfrom = dfVfrom.iloc[:,1:nc].to_numpy()
        #vto = dfVto.iloc[:,1:nc].to_numpy()
        
        # -- Step 3.: Time-vector --
        def extractTimeVectorFromDataframeV3(myDf, coltimeName, resetDunResetIndex, convertDate, convertDfDateTo64):
            # resetDunResetIndex = ["reset index", "reset"] --> ...
            #    ... does the user want an index or not    
            if resetDunResetIndex=="reset index" or resetDunResetIndex=="reset": 
                myDf = myDf.reset_index()
            #  convertDate = ["datetime64" , "datetime64[D]", "datetime64[ns]" ==> ...
            #    ... convert date 
            # note: for .e.g, for the crypto daily data from CryptoCompare the format is
            # '2017-08-31T00:00:00.000000000' when the reader fetch csv data.
            # to transform this fomrat into '2017-08-31', the user specifies
            # convertDate = "datetime64[D]"
            if convertDate == 'datetime64' or convertDate == 'datetime64[ns]'\
                or convertDate == 'datetime64[D]':
                benchDate_np = myDf[coltimeName].to_numpy(dtype=convertDate)        
            else:
                benchDate_np = myDf[coltimeName].to_numpy()
            # As above for the dataFrame format
            if convertDfDateTo64 == 'datetime64' or convertDfDateTo64 == 'datetime64[ns]'\
                or convertDfDateTo64 == 'datetime64[D]':
                benchDate_df = myDf[coltimeName].astype(convertDfDateTo64)        
            else:
                benchDate_df = myDf[coltimeName]     
            # Return output
            return benchDate_df , benchDate_np        
        
        benchdate_df, benchdate_np = extractTimeVectorFromDataframeV3(\
                    dfC, "time", "dunresetIndex", "datetime64[D]", 'datetime64[ns]')        
        
        # -- Step 4.: Assign sub-universe --
        # Dataframe format
        self.dfO = dfO
        self.dfH = dfH
        self.dfL = dfL
        self.dfC = dfC
        self.dfVfrom = dfVfrom
        self.dfVto = dfVto
        self.benchdate_df = benchdate_df
        self.benchdate_np = benchdate_np        
        # Numpy array format
        #self.o = o
        #self.h = h
        #self.l = l
        #self.c = c
        #self.vfrom = vfrom
        #self.vto = vto    
        
#______________________________________________________________________________        
class dataStructureCsv_fromFullUniverse():
    
    # histDataDir_fullUniverse = chosenDirectory

    # -- INITIALISIATION --
    def __init__(self, ccy1=None, chosenDirectory=None,\
                 dfO=None, dfH=None, dfL=None, dfC=None, \
                 dfVfrom=None, dfVto=None):
        # Input
        self.ccy1 = ccy1
        self.chosenDirectory = chosenDirectory
        # OHLCV dataframe
        #self.dfO = dfO
        #self.dfH = dfH
        #self.dfL = dfL
        #self.dfC = dfC
        #self.dfVfrom = dfVfrom
        #self.dfVto = dfVto
        
    # -- HISTORICAL DATA --
    def getCsvHistoricalData(self, ccy1, chosenDirectory):
        
        print(chosenDirectory)
        timeCcyColHeaders = ccy1.copy()
        timeCcyColHeaders.insert(0, 'time')

        # -- Step 1.: Extract the sub-universe --
        # Open
        dfO = pd.read_csv(chosenDirectory + "openPrice.csv", header=0, sep=',', parse_dates=['time'])
        dfO = dfO[timeCcyColHeaders]
        dfO = dfO.set_index(['time'])
        dfO = dfO.reset_index()
        #dfO['time'] = pd.to_datetime(dfO['time'])
        # High
        dfH = pd.read_csv(chosenDirectory + "highPrice.csv", header=0, sep=',', parse_dates=['time'])
        dfH = dfH[timeCcyColHeaders]
        dfH = dfH.set_index(['time'])
        dfH = dfH.reset_index()
        #dfH['time'] = pd.to_datetime(dfH['time'])
        # Low
        dfL = pd.read_csv(chosenDirectory + "lowPrice.csv", header=0, sep=',', parse_dates=['time'])
        dfL = dfL[timeCcyColHeaders]
        dfL = dfL.set_index(['time'])
        dfL = dfL.reset_index()      
        #dfL['time'] = pd.to_datetime(dfL['time'])        
        # Close
        dfC = pd.read_csv(chosenDirectory + "closePrice.csv", header=0, sep=',', parse_dates=['time'])
        dfC = dfC[timeCcyColHeaders]
        dfC = dfC.set_index(['time'])
        dfC = dfC.reset_index() 
        #dfC['time'] = pd.to_datetime(dfC['time'])          
        # Close dfV (Vfrom)
        dfVfrom = pd.read_csv(chosenDirectory + "volumeFrom.csv", header=0, sep=',', parse_dates=['time'])
        dfVfrom = dfVfrom[timeCcyColHeaders]
        dfVfrom = dfVfrom.set_index(['time'])
        dfVfrom = dfVfrom.reset_index()  
        #dfV['time'] = pd.to_datetime(dfV['time'])
        # Close dfVto
        dfVto = pd.read_csv(chosenDirectory + "volumeTo.csv", header=0, sep=',', parse_dates=['time'])
        dfVto = dfVto[timeCcyColHeaders]
        dfVto = dfVto.set_index(['time'])
        dfVto = dfVto.reset_index()     
        #dfVto['time'] = pd.to_datetime(dfVto['time'])

        # -- Step 3.: Time-vector --
        def extractTimeVectorFromDataframeV3(myDf, coltimeName, resetDunResetIndex, convertDate, convertDfDateTo64):
            # resetDunResetIndex = ["reset index", "reset"] --> ...
            #    ... does the user want an index or not    
            if resetDunResetIndex=="reset index" or resetDunResetIndex=="reset": 
                myDf = myDf.reset_index()
            #  convertDate = ["datetime64" , "datetime64[D]", "datetime64[ns]" ==> ...
            #    ... convert date 
            # note: for .e.g, for the crypto daily data from CryptoCompare the format is
            # '2017-08-31T00:00:00.000000000' when the reader fetch csv data.
            # to transform this fomrat into '2017-08-31', the user specifies
            # convertDate = "datetime64[D]"
            if convertDate == 'datetime64' or convertDate == 'datetime64[ns]'\
                or convertDate == 'datetime64[D]':
                benchDate_np = myDf[coltimeName].to_numpy(dtype=convertDate)        
            else:
                benchDate_np = myDf[coltimeName].to_numpy()
            # As above for the dataFrame format
            if convertDfDateTo64 == 'datetime64' or convertDfDateTo64 == 'datetime64[ns]'\
                or convertDfDateTo64 == 'datetime64[D]':
                benchDate_df = myDf[coltimeName].astype(convertDfDateTo64)        
            else:
                benchDate_df = myDf[coltimeName]     
            # Return output
            return benchDate_df , benchDate_np        
        
        benchdate_df, benchdate_np = extractTimeVectorFromDataframeV3(\
                    dfC, "time", "dunresetIndex", "datetime64[D]", 'datetime64[ns]')        
        
        # -- Step 4.: Assign sub-universe --
        # Dataframe format
        self.dfO = dfO
        self.dfH = dfH
        self.dfL = dfL
        self.dfC = dfC
        self.dfVfrom = dfVfrom
        self.dfVto = dfVto
        self.benchdate_df = benchdate_df
        self.benchdate_np = benchdate_np        
    