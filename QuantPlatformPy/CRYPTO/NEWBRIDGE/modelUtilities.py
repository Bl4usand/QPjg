# -*- coding: utf-8 -*-
"""
CONFIGURATION FILE
"""
#______________________________________________________________________________
# IMPORT PACKAGES, DEFINE PATHS, IMPORT API-KEYS
# -- Import packages --
import importlib, importlib.machinery
import sys
# import setuptools, imp, csv
import pandas as pd
#import pandas_datareader.data as pdr
import numpy as np
#from numpy import inf
#import datetime as dt
#from datetime import datetime  # , timtedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
# import cryptocompare as cryc # NO NEED!!!! request the url ofthe APi !!!!!!
#import requests
#import inspect

#______________________________________________________________________________
# Add the different dependencies
#def manage_model_dependencies(masterPath):

    # -- Add more utilities --
    #sys.path.insert(0,masterPath+'generalUtilities')  
    #sys.path.insert(0,masterPath+'dataVisualization')   
    #sys.path.insert(0,masterPath+'statisticalFunctions') 
    #sys.path.insert(0,masterPath+'portfolioConstruction') 
    #sys.path.insert(0,masterPath+'backtestUtilities')
    
    # Step 2: Import desired functions
    #mantime = importlib.import_module('manage_time', __name__)
    #npu     = importlib.import_module('NumpyArrays_Utilities', __name__)
    #dfu     = importlib.import_module('PandaDf_Utilities', __name__)
    #linch   = importlib.import_module('Line_Charts1', __name__)
    #stfun   = importlib.import_module('statFunctions', __name__)
    #volptf  = importlib.import_module('VolatilityBased_Portfolio', __name__)
    #eqsptf  = importlib.import_module('EqualSharpe_Portfolio', __name__)
    #futbu   = importlib.import_module('FuturesReturns_Utilities', __name__)
    #futptbu = importlib.import_module('futures_PortfolioReturns_Utils', __name__) 
    #mvow    = importlib.import_module('MVO_StaticWrapper', __name__)
    #ercw    = importlib.import_module('ERC_Wrapper', __name__)
    #hrpw    = importlib.import_module('HRP_Wrapper', __name__)

    # Step 3:
    #return mantime, npu, dfu, linch, stfun, volptf, eqsptf, futbu,\
    #    futptbu, mvow, ercw, hrpw

mainDrive = '/home/dkuenzi/PycharmProjects/QPjg/'
masterDir = 'QuantPlatformPy/'
projectDir = 'CRYPTO/NEWBRIDGE/'

# ______________________________________________________________________________
# Add Paths for algorithmic controlers, utilities, universes
def addPath(mainDrive, masterDir, projectDir):
    mainDrive = mainDrive
    masterDir = masterDir  # 'QuantPlatformPy/'
    masterPath = mainDrive + masterDir
    projectDir = projectDir  # 'econometricAnalysis/'
    projectPath = masterPath + projectDir
    return mainDrive, masterDir, masterPath, projectDir, projectPath

# ______________________________________________________________________________
# Configure model
def modelConfiguration(universeNb, masterDir, projectDir):
    
    # General parameters
    exchange = 'coinbase'
    timeframe = 'day'
    nbOfObs =  7*256 # nb. of of data points downloaded from CryptoCompare
    # note: Use 7*256 as default for "donwload CC data"
    #       If "donwload CC data & concatenateCSV", then nbOfObs=15 in the code

    # -- Directory for output (make it universe dependent) --
    #outputDir   = "F:/" + masterDir+projectDir + "IODY_2RT/modelOutput/"
    #histDataDir = "F:/" + masterDir+projectDir + "IODY_2RT/data_historical/"
    
    # -- Directory for historical full universe output --
    universeFullPath = '/home/dkuenzi/PycharmProjects/QPjg/QuantPlatformPy/CRYPTO/NEWBRIDGE/fullUniverse/fullUniverse.csv'
    dataDir_fullUniverse = '/home/dkuenzi/PycharmProjects/QPjg/QuantPlatformPy/CRYPTO/NEWBRIDGE/fullUniverse/data/'
    histDataDir_fullUniverse = '/home/dkuenzi/PycharmProjects/QPjg/QuantPlatformPy/CRYPTO/NEWBRIDGE/fullUniverse/historicalData/'

    # Select universe
    if universeNb == "baseUniverse":
        ccy1 = ['BTC', 'ETH']
        ccy2 = ['USD' for c in ccy1]
        # Transaction cost in basis points
        instTC = np.array([10, 15])
        # Budgets' constraints for risk budgeting
        budgets = [40, 21]    
        
    elif universeNb == "fullUniverse":
        # Universe
        #df_fullUniverse = pd.read_csv(universeFullPath , index_col=0)
        #df_fullUniverse = df_fullUniverse.reset_index()
        df_fullUniverse = pd.read_csv(universeFullPath)
        #nb_of_insts = len(df_universe['symbol'])
        # Universe
        df_subFullUniverse = df_fullUniverse.copy() 
        df_subFullUniverse = df_subFullUniverse.loc[df_subFullUniverse['fullUniverseInclusion']==1]
        # Extract the tickers
        symbolsList = df_subFullUniverse['symbol'].tolist()        
        sectorsList = df_subFullUniverse['sector'].tolist() 
        tcList = df_subFullUniverse['transCost'].tolist() 
        
        # THIS IS COMPLETELY USELESS HERE....
        outputDir   = mainDrive + masterDir+projectDir + "IODY_2RT/modelOutput/"
        histDataDir = mainDrive + masterDir+projectDir + "IODY_2RT/data_historical/"

        #  rename
        ccy1 = symbolsList
        ccy2 = ['USD' for c in ccy1]
        # Transaction cost in basis points
        instTC = np.asarray(tcList)
        # Budgets' constraints for risk budgeting
        budgets = 0*[40, 21, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]      
        # Sector
        sector = sectorsList
        
    elif universeNb == "coreUniverse":
        #  NEAR, LUNA
        ccy1 = ['BTC',  'ETH',\
                'BNB',  'DOT',  'ADA',   'SOL',\
                'AVAX', 'ATOM', 'MATIC', 'NEAR',\
                'ALGO', 'LINK', 'FIL', 'UNI',\
                'XRP',  'XMR',  'XLM']
        ccy2 = ['USD' for c in ccy1]
        # Sector
        sector = ['currency',        'smartContract',\
                  'smartContract',   'smartContract', 'smartContract',\
                  'l2smartContract', 'smartContract', 'smartContract',\
                  'smartContract',   'smartContract', 'smartContract',\
                   'smartContract',  'web3_dataManagement', 'storage', 'defi',\
                  'currency', 'currency', 'currency']
        # Transaction cost in basis points
        instTC = np.array([10, 15, 20, 20, 20, 20, 20, 20, 20, 20, 25, 25,25,\
                           25, 20, 20, 25])
        # Budgets' constraints for risk budgeting
        budgets = [40, 21, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  

        # -- Directory for output (make it universe dependent) --
        outputDir   = mainDrive+masterDir+projectDir + "modelOutput/"
        histDataDir = mainDrive+masterDir+projectDir + "fullUniverse/historicalData"

    # AUM
    aum = 1000000
    # Target volatility
    tgtVolatility = 25/100

    # Return        
    return exchange, timeframe, nbOfObs, ccy1, ccy2, instTC,\
        aum, budgets, tgtVolatility, outputDir, histDataDir,\
        dataDir_fullUniverse, histDataDir_fullUniverse

#______________________________________________________________________________
# Import user's package
def addUserPackage(masterPath, packageDirName, packageName, packageCode):
    # Example: packageDirName = 'generalUtilities'
    #          packageName    = 'manage_time'
    #          packageCode    = mantime
    # step 1: Insert paths of user's packages
    sys.path.insert(0, masterPath + packageDirName)
    # Step 2: Import desired functions
    packageCode = importlib.import_module(packageName, __name__)
    return packageCode

#______________________________________________________________________________
# Line for flat matrices of OHLC from Cryptocompare
def plottsv1(myDf, ccyname):
    # - Extract data --
    mydata = pd.DataFrame({'time': myDf['time'], ccyname: myDf[ccyname]})
    # -- Define figure --
    fig, ax = plt.subplots(figsize=(12, 6))
    # -- Define style --
    sns.set_style("darkgrid")
    # -- Create figure --
    sns.lineplot(data=mydata, x="time", y=ccyname, color='black', linewidth=1)
    plt.ylabel("close price", size=10)
    dateSolution = 2
    if dateSolution == 1:
        plt.xlabel("", size=8)
    else:
        x_dates = mydata['time'].dt.strftime('%Y-%m-%d').sort_values().unique()
        ax.set_xticklabels(labels=x_dates, rotation=45, ha='right', size=8)
        plt.xlabel("", size=8)
    plt.title(ccyname)
    # plt.legend()

#______________________________________________________________________________
def plotCorrMatrix(retunrsDf, method):
    # Corr matrix dataframe
    corr_matrix = retunrsDf.corr()

    if method == 1:
        # Create figure
        fig = plt.figure(figsize=(19, 15))
        plt.matshow(corr_matrix, fignum=fig.number)
        plt.xticks(range(retunrsDf.shape[1]), retunrsDf.columns, fontsize=14, rotation=45)
        plt.yticks(range(retunrsDf.shape[1]), retunrsDf.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16);

    elif method == 2:
        # Create figure
        # fig = plt.figure(figsize=(19, 15))
        corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
        # plt.matshow(corr_matrix, fignum=fig.number)
        # plt.xticks(range(retunrsDf.shape[1]), retunrsDf.columns, fontsize=14, rotation=45)
        # plt.yticks(range(retunrsDf.shape[1]), retunrsDf.columns, fontsize=14)
        # cb = plt.colorbar()
        # cb.ax.tick_params(labelsize=14)
        # plt.title('Correlation Matrix', fontsize=16);

        fig, ax = plt.subplots(figsize=(19, 15))
        ax.matshow(corr_matrix)
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns);
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns);

    elif method == 3:
        corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).set_properties(**{'font-size': '0pt'})

    return corr_matrix

#______________________________________________________________________________
# PEPARE DATA FOR BACKTEST
def prepareDataForBacktest(crydat):
    dfO = crydat.dfO
    dfH = crydat.dfH
    dfL = crydat.dfL
    dfC = crydat.dfC
    [nn, nc] = dfC.shape
    o = dfO.iloc[:, 1:nc].to_numpy()
    h = dfH.iloc[:, 1:nc].to_numpy()
    l = dfL.iloc[:, 1:nc].to_numpy()
    c = dfC.iloc[:, 1:nc].to_numpy()
    return o, h, l, c

#______________________________________________________________________________
# WRITE CSV

def write_outputCsv(downloadDataOption, outputDir, histDataDir, \
                    dataStructure, plDf, ptfplClass,\
                    benchdate_np, ccy1):
    
    # -- Always write CSV to outputDir by default -----------------------------
    # Open price
    csvName = 'openPrice.csv'
    csvPath = outputDir + csvName
    dataStructure.dfO.to_csv(csvPath, index=False)
    # High price
    csvName = 'highPrice.csv'
    csvPath = outputDir + csvName
    dataStructure.dfH.to_csv(csvPath, index=False)    
    # Low price
    csvName = 'lowPrice.csv'
    csvPath = outputDir + csvName
    dataStructure.dfL.to_csv(csvPath, index=False)
    # Close price
    csvName = 'closePrice.csv'
    csvPath = outputDir + csvName
    dataStructure.dfC.to_csv(csvPath, index=False)   
    # Volume to
    csvName = 'volumeTo.csv'
    csvPath = outputDir + csvName
    dataStructure.dfVto.to_csv(csvPath, index=False)   
    # Volume from
    csvName = 'volumeFrom.csv'
    csvPath = outputDir + csvName
    dataStructure.dfVfrom.to_csv(csvPath, index=False)   
    # Strategy return
    csvName = 'stratreturn.csv'
    csvPath = outputDir + csvName
    plDf.to_csv(csvPath, index=False)
    # Weights
    #wgtDf = pd.DataFrame(data=ptfplClass.wgt, index=benchdate_np, columns=ccy1)
    signedWeightsDf = pd.DataFrame(data=ptfplClass.signedWeights, index=benchdate_np, columns=ccy1)
    csvName = 'signedWeights.csv'
    csvPath = outputDir + csvName
    signedWeightsDf.to_csv(csvPath, index=False) 
    
    # Weight sum table
    btcWgt = ptfplClass.signedWeights[-1,0]
    ethWgt = ptfplClass.signedWeights[-1,1]
    altWgt = np.sum(ptfplClass.signedWeights[-1,2:])
    cashWgt = 1- np.sum(ptfplClass.signedWeights[-1,:])   
    tableWgt = [['currency', 'amount'], ['BTC', btcWgt], ['ETH', ethWgt ],\
             ['Alt-Coins', altWgt], ['Cash', cashWgt]]
    #print(tabulate(tableWgt))    
    #print(tabulate(tableWgt, headers='firstrow'))
    print(tabulate(tableWgt, headers='firstrow', tablefmt='fancy_grid'))
    # List of ccies and weights
    ccy = ['BTC', 'ETH', 'Alt-Coins', 'Cash']
    ccywgt = [btcWgt, ethWgt, altWgt, cashWgt]
    # Dictionary of lists
    Tdict = {'currency': ccy, 'amount': ccywgt}
    Tdf = pd.DataFrame(Tdict) 
    # saving the dataframe 
    Tdf.to_csv('F:/R2T_INCUTRUST/DATA/'+'allocation.csv', index=False) 
    
    # -- Write CSV to histDataDir if only if downloadDataOption=
    # "donwload CCdata_concatenateCSV"                                      ---
    
    if downloadDataOption == "donwload CC data & concatenateCSV":
       # Open price
       csvName = 'openPrice.csv'
       csvPath = histDataDir + csvName
       dataStructure.dfO.to_csv(csvPath, index=False)
       # High price
       csvName = 'highPrice.csv'
       csvPath = histDataDir + csvName
       dataStructure.dfH.to_csv(csvPath, index=False)    
       # Low price
       csvName = 'lowPrice.csv'
       csvPath = histDataDir + csvName
       dataStructure.dfL.to_csv(csvPath, index=False)
       # Close price
       csvName = 'closePrice.csv'
       csvPath = histDataDir + csvName
       dataStructure.dfC.to_csv(csvPath, index=False)   
       # Volume to
       csvName = 'volumeTo.csv'
       csvPath = histDataDir + csvName
       dataStructure.dfVto.to_csv(csvPath, index=False)   
       # Volume from
       csvName = 'volumeFrom.csv'
       csvPath = histDataDir + csvName
       dataStructure.dfVfrom.to_csv(csvPath, index=False)   
       # Strategy return
       csvName = 'stratreturn.csv'
       csvPath = histDataDir + csvName
       plDf.to_csv(csvPath, index=False)
       # Weights
       #wgtDf = pd.DataFrame(data=ptfplClass.wgt, index=benchdate_np, columns=ccy1)
       signedWeightsDf = pd.DataFrame(data=ptfplClass.signedWeights, \
                                      index=benchdate_np, columns=ccy1)
       csvName = 'signedWeights.csv'
       csvPath = histDataDir + csvName
       signedWeightsDf.to_csv(csvPath, index=False)              
    
# -- Compute Drawdown ---------------------------------------------------------  
def computeDrawdown(cumret):
    # -- Preallocation --
    nbobs = cumret.size
    highwatermark    = np.zeros((nbobs))  # initialize high watermarks to zero.
    drawdown         = np.zeros((nbobs))  # initialize drawdowns to zero.
    drawdownduration = np.zeros((nbobs))  # initialize drawdown duration to zero.
    # -- Main loop --
    for t in range(nbobs):
        highwatermark[t] = max(highwatermark[t-1], cumret[t])
        drawdown[t] = (1 + cumret[t] ) / (1 + highwatermark[t]) - 1 # drawdown on each day
        if drawdown[t] == 0:
            drawdownduration[t] = 0;
        else:
            drawdownduration[t] = drawdownduration[t-1] + 1
    maxDD  = np.min(drawdown)         # maximum drawdown
    maxDDD = np.max(drawdownduration) # maximum drawdown duration
    return maxDD, maxDDD

def computeDrawdownTS(cumret, cumret_type):
    # -- Preallocation --
    nbobs = cumret.size
    tsdd = np.zeros((nbobs))   
    # -- Compute returns --
    if cumret_type == 'geometric':
        returns = np.zeros((nbobs)) 
        Y = np.zeros((nbobs)) 
        for i in range(1,nbobs):
            if cumret[i-1] != 0:
                returns[i] = cumret[i] / cumret[i-1] - 1
        Y = np.cumsum(returns) # Cumulated returns
    elif cumret_type == 'arithmetic':
        Y = cumret.copy() # Cumulated returns
    # -- Drawdown time series --
    for i in range(nbobs):
        # note:  Y[1:i+1] ---> vector size i+1-1, same final index than Y[i]
        tsdd[i] = np.max(Y[0:i+1])-Y[i]
    # -- Max drawdown --
    maxdd = np.max(tsdd)
    maxddIdx= tsdd.argmax(axis=0) #np.where(tsdd == maxdd)
    return tsdd, maxdd, maxddIdx
#------------------------------------------------------------------------------ 

# ---- Identify cycle for rebalancing -----------------------------------------
# note: this is used to identify ed of week, month, quarter, semester as in my
# Matlab's environment
#
def calculationCycle(dateBench, objectType, method):
    # Transform the panda date index into an array
    if objectType=="series" or objectType=="Series":
        dateArray = dateBench
    else:
        dateArray = dateBench.to_pydatetime()
    #dateArray = dateBench.date()
    # Pre-allocation
    n = dateBench.shape[0]
    estimationCycle = np.zeros(n)
    if method == 'w' or method == 'weekly':
        for i in range(1,n):
        #for i in  (number+1 for number in range(n-1)):
            tprev = dateBench.iloc[i-1]
            tcur = dateBench.iloc[i]
            if tprev.week != tcur.week:
                estimationCycle[i-1] = 1     
    elif method == 'm' or method == 'monthly':
        for i in range(1,n):
            tprev = dateBench.iloc[i-1]
            tcur = dateBench.iloc[i]
            if tprev.month != tcur.month:
                estimationCycle[i-1] = 1      
    elif method == 'q' or method == 'quarterly':
        for i in range(1,n):
            tprev = dateBench.iloc[i-1]
            tcur = dateBench.iloc[i]
            if tprev.month != tcur.month and np.mod(tprev.month,3)==0:
                estimationCycle[i-1] = 1  
    elif method == 'hy' or method == 'halfyearly':
        for i in range(1,n):
            tprev = dateBench.iloc[i-1]
            tcur = dateBench.iloc[i-1]
            if tprev.month != tcur.month and np.mod(tprev.month,6)==0:
                estimationCycle[i-1] = 1  
    elif method == 'y' or method == 'yearly':
        for i in range(1,n):
            tprev = dateBench.iloc[i-1]
            tcur = dateBench.iloc[i]
            if tprev.year != tcur.year:
                estimationCycle[i-1] = 1   
    # Total number of computations "top"
    estimationCycleTot = sum(estimationCycle)                   
    return estimationCycle, dateArray, estimationCycleTot 
#------------------------------------------------------------------------------  

# -- Rolling Delta Function ---------------------------------------------------
def Delta(x, method, parameters):
    # -- Dimensions -----------------------------------------------------------
    try:
        (nsteps,ncols) = x.shape
    except:
        x = x.reshape(-1,1)
        tempDim = x.shape
        nsteps = tempDim[0]
        ncols = 1
    # Prelocate output
    y = np.zeros((nsteps,ncols))
    # -- Rate of change -------------------------------------------------------
    if method == "r" or method == "roc" or method == "ROC" or method == "rate of change":
        lookback = parameters[0]
        for j in range(ncols):
            xj = x[:,j]
            for i in range(lookback,nsteps):
                if xj[i-lookback]!=0:
                    y[i,j] = xj[i]/xj[i-lookback]-1
    # -- Difference -----------------------------------------------------------            
    elif method == "d" or method == "dif" or method == "difference":
        lookback = parameters[0]
        for j in range(ncols):
            xj = x[:,j]
            for i in range(lookback,nsteps):
                y[i,j] = xj[i]-xj[i-lookback]
    # -- Output ---------------------------------------------------------------                
    return y
#------------------------------------------------------------------------------  

# -- Rolling standard deviation -----------------------------------------------
def rollingVolatility(x,lookback):
    # Dimensions
    try:
        (nsteps,ncols) = x.shape
        #[nsteps,ncols] = x.shape
    except:		
        x = x.reshape(-1,1)
        tempDim = x.shape
        nsteps = tempDim[0]
        ncols = 1
        #nsteps = x.size
        #ncols = 1
    # Pre-allocation
    #z = np.zeros([nsteps,ncols])
    z = np.zeros((nsteps,ncols))
    try:
        lookback = lookback.astype(int)
    except:
        lookback = lookback    
    # Main loop
    for j in range(ncols):
        xj = x[:,j]
        for i in range(lookback,nsteps):
            # note: xSnap = xj[i-lookback:i] gives vector till 'i-1'th element
             #      if this convention is used, result has to be assigned to z[i-1,j]
            xSnap = xj[i-lookback+1:i+1] # I checked with Matlab results
            volSnap = np.std(xSnap,axis=0, dtype = np.float64, ddof=1)
            z[i,j] = volSnap
    # -- Output --               
    return z
#------------------------------------------------------------------------------  


# -- Charting tool ------------------------------------------------------------
def plotWrapper_Format1(df,dateName,colName,myColor,myTitle,plotOption):
    
    if plotOption == 'option1':
        ax = plt.gca()
        df.plot(kind='line',x=dateName,y=colName, color=myColor,title=myTitle,ax=ax)
        plt.show()  
    elif plotOption == 'option2':
        sns.set(style="darkgrid")
        ax = sns.lineplot(x=dateName, y=colName,data=df)
        plt.title(myTitle)
    elif plotOption == 'option3':
        ax = sns.relplot(x=dateName, y=colName,kind="line",data=df)
        ax.fig.autofmt_xdate()    
        plt.title(myTitle)
#------------------------------------------------------------------------------        


#______________________________________________________________________________
#
# Preallocation for future-level P&L
#______________________________________________________________________________


def futureLevel_PreallocationForPL(x):
    (nrow,ncols) = x.shape
    grossreturn = np.zeros((nrow,ncols))  
    tcforec = np.zeros((nrow,ncols)) 
    return grossreturn, tcforec

#______________________________________________________________________________
#
# Function to Compute P&L at the stock / Future level
# Step 1 adjust the transaction cost
# Step 2 compute the P&L
#______________________________________________________________________________
#

# Note 1: as in Matlab, the P&L is always realized at next bar open based
#         on signal and weight computed at the previous bar close: 
#    
#......close @ i-2......open @ i-1......open @ i..............close @ i.... 
#     signal & weight   execution       P&L realised  ------> open-based-return[i]
#                                       P&L observed at Open, blind to close  
#
#       open-based-return[i] = signal[i-2] * weight[i-2] * (open[i]/open[i-1]-1)
#..........................................................................      
#        This P&L is the one realized in real life as I execute at next bar open.
#        If I code it this way, for a daily time series, the return the code
#        would show in step "i" has a close-to-open time delay. 
#        Nothing wrong with this, but I chose another convention to show P&L.

# Noe 2: By CONVENTION, I chose to visualize the P&L at the end of bar "i"
#        using the open at bar "i+1".
#        There IS NOT ANY FORWARD-LOOKING BIAS as, if the P&L behaviour is
#        factored in to feedback into the algorithm, proper lag is applied.
#        To do so, I define an execution price p as nextbar open such as:
#                              p[i] = open[i-1]
#
#          VISUALISING P&L AT THE END OF BAR "i" WITH OPEN "i+1"
#          means, compared to previous solution, open-based-return[i]
#          is oberved at the close at step/bar "i-1"
#
#      !!! THE CODE OBSERVE open-based-return[i] AT CLOSE AT TIME "i-1" !!!
# 
#......close @ i-1.........p @ i-1.........p @ i..............close @ i.... 
#                          open @ i      open @ i+1  <--- !!!!!!!!!
#     signal & weight        (p[i,j]/p[i-1,j]-1) ------> ObservedReturn[i]
#                            is the return oberved tomorrow morning (or at next bar open "i+1")
#                            and shown at the end of day/bar "i". 
#
#      ObservedReturn[i] = s[i-1,j] * wgt[i-1,j] * (p[i,j]/p[i-1,j]-1)
#      ObservedReturn[i] = s[i-1,j] * wgt[i-1,j] * (open[i+1,j]/open[i,j]-1)
#             ObservedReturn[i] = open-based-return[i+1] <--- my convention
#..........................................................................  

def Compute_StockFuture_PL(i, j, c, p, ExecP, s, wgt, nb, TC):
    
    #################### UBER IMPORTANT ##############
    lagSignal = 1
    ################################################    
    
    # -- Step 1: Process to compute transaction Cost --------------------------
    # Preset the variables
    Ftc1=0
    Ftc2=0
    Ftc3=0
    # Ftc1 = Factor when Trade Out
    # Ftc2 = Factor when Trade In
    # Ftc3 = Factor when Nb of shares is different for same signals
    if s[i-1,j] == s[i-2,j]:
        Ftc1 = 0 # Factor when Trade Out
        Ftc2 = 0 # Factor when Trade In
    elif s[i-2,j] == 0 and s[i-1,j] != 0:
        Ftc1 = 0 # Factor when Trade Out
        Ftc2 = 1 # Factor when Trade In
    elif s[i-2,j] != 0 and s[i-1,j] == 0:
        Ftc1 = 1 # Factor when Trade Out
        Ftc2 = 0 # Factor when Trade In
    elif (s[i-2,j] == 1 and s[i-1,j] == -1) or (s[i-2,j] == -1 and s[i-1,j] == 1):
        Ftc1 = 1 # Factor when Trade Out
        Ftc2 = 1 # Factor when Trade In
    # -- Ftc3 for nb.-of-shares-based P&L --
    if s[i,j] != s[i-1,j] or (s[i,j] == s[i-1,j] and nb[i,j] == nb[i-1,j]):
        Ftc3 = 0
    elif (s[i,j] == s[i-1,j] and nb[i,j] != nb[i-1,j]):
        Ftc3 = 1
    else:
        Ftc3 = 0

    # -- Ftc3 for weights-based equity curve --
    if s[i,j] != s[i-1,j] or (s[i,j] == s[i-1,j] and wgt[i,j] == wgt[i-1,j]):
        Ftc3wgt = 0
    elif (s[i,j] == s[i-1,j] and wgt[i,j] != wgt[i-1,j]): 
        Ftc3wgt = 1
    else:
        Ftc3wgt = 0

    # -- Step 2: Stock/Future P&L ---------------------------------------------
    
    if ~np.isnan(p[i,j]) and ~np.isnan(p[i-1,j]) and p[i,j] != 0 and p[i-1,j] != 0:
        
        #grossreturn_i = -0.5+np.random.rand()
        #grossreturn_i = (p[i,j]/p[i-1,j]-1)
        #grossreturn_i =  wgt[i-lagSignal,j] * (p[i,j]/p[i-1,j]-1)
        grossreturn_i = s[i-lagSignal,j] * wgt[i-lagSignal,j] * (p[i,j]/p[i-1,j]-1)
        #grossreturn_i = 1 * wgt[i-lagSignal,j] * (p[i,j]/p[i-1,j]-1)
        
        tcforec_i = Ftc1 * TC[j] * wgt[i-2,j] + Ftc2 * TC[j] * wgt[i-1,j] +\
            Ftc3wgt*TC[j]*abs(wgt[i-1,j]-wgt[i-2,j]) # wgt or nb  
        
    else:
        
        grossreturn_i = 0
        tcforec_i = 0
        
    return grossreturn_i, tcforec_i

def  ChoseExecutionPrice(o,h,l,c,vwap, method, OpenWgt):    
    # -- Dimensions & Prelocate matrices --
    (nsteps,ncols) = c.shape;
    p = np.zeros((nsteps,ncols))

    if method == "open":
        p[0:nsteps-2,:] = o[1:nsteps-1,:]
        p[nsteps-1,:] = p[nsteps-2,:]
    elif method == "atp":
        p[0:nsteps-2,:] = (o[1:nsteps-1,:]+h[1:nsteps-1,:]+l[1:nsteps-1,]+c[1:nsteps-1,:])/4
        p[nsteps-1,:] = p[nsteps-2,:]
    elif method == "atp_open":
        open_weight = OpenWgt;
        p[0:nsteps-2,:] = open_weight*o[1:nsteps-1,:] + (1-open_weight)/3*(h[1:nsteps-1,:] + l[1:nsteps-1,:] + c[1:nsteps-1,:]);
        p[nsteps-1,:] = p[nsteps-2,:];
    elif method == "vwap":
        p[0:nsteps-2,:] = vwap[1:nsteps-1,:]; 
        p[nsteps-1,:] = p[nsteps-2,:];      
    elif method == "vwap_open":
        open_weight = OpenWgt;
        p[0:nsteps-2,:] = open_weight*o[1:nsteps-1,:]+(1-open_weight)*vwap[1:nsteps-1,:];
        p[nsteps-1,:] = p[nsteps-2,:]; 
    # -- return outuput --
    return p

def ptfOfFuturesLevel_PreallocationForPL(x, aum, assetClassTC):

    # -- Dimensions --
    try:
        (nsteps, ncols) = x.shape
    except:
        ncols=1
        nsteps=x.shape
            
    # -- Portfolio volatility --
    # ptfret=zeros(size(c,1),1);
    # volptf=zeros(size(c,1),1);
    nb          = np.zeros((nsteps,ncols))       # Number of Shares
    s           = np.zeros((nsteps,ncols))       # Signals
    wgtGross    = np.zeros((nsteps,ncols))       # Gross Weights
    wgt         = np.zeros((nsteps,ncols))       # Weights   
    ExecP       = np.zeros((nsteps,ncols))       # Execuion price
    grossreturn = np.zeros((nsteps,ncols))
    tcforec     = np.zeros((nsteps,ncols))
    
    # -- Prelocation --
    
    # gross
    netret           = np.zeros((nsteps,ncols))
    cumulgrossret    = np.zeros((nsteps,ncols))
    ptfecgross       = 100*np.ones((nsteps,1))
    ptfplgross       = 100*np.ones((nsteps,1)) 
    stratreturngross = np.zeros((nsteps,1))
    instGeoPlTemp    = 100*np.ones((nsteps,ncols))
    
    # net
    cumulnetret      = np.zeros((nsteps,ncols))
    ptfec            = 100*np.ones((nsteps,1))
    ptfpl            = 100*np.ones((nsteps,1))
    stratreturn      = np.zeros((nsteps,1))
    capital          = aum
    capitalGeo       = np.zeros((nsteps,1))
    capitalGeo[0]    = aum;

    TC               = assetClassTC * 0.0001  # Transaction cost
    anVolat          = np.zeros((nsteps,1))
    assetReturns     = Delta(x, 'roc', np.array([1]))        # Compute returns
    adjustedweight   = np.zeros((nsteps,ncols))  #
    maxLevj          = np.zeros((1,ncols))       # Initialisation
    adjFactorj       = np.zeros((1,ncols)) 

    # -- Return output --
    return nsteps, ncols, nb, s, wgtGross, wgt, ExecP, grossreturn, tcforec, netret, cumulgrossret, ptfecgross, ptfplgross, stratreturngross,  instGeoPlTemp, cumulnetret, ptfec, ptfpl, stratreturn, capital, capitalGeo, TC, anVolat, adjustedweight, maxLevj, adjFactorj, assetReturns


#
#__________________________________________________________________________
#
# Compute Performance of the portoflio
#__________________________________________________________________________
#

def timeUnitPtfPerf(rowIndex, netret, grossreturn, tcforec, cumulnetret, \
                    ptfecgross, ptfplgross, stratreturn, stratreturngross, ptfec, ptfpl):


    netretT = grossreturn[rowIndex,:] - tcforec[rowIndex,:]
    
    # -- Compute Ptf return--
    cumulnetretT = cumulnetret[rowIndex-1,:] + netret[rowIndex,:]
    cumulgrossretT = grossreturn[rowIndex-1,:] + netret[rowIndex,:]#Useless
    
    # gross
    ptfecgrossT = ptfecgross[rowIndex-1] * \
        (1 + np.sum(grossreturn[rowIndex,:]))
    ptfplgrossT = ptfplgross[rowIndex-1] + \
        (100 * np.sum(grossreturn[rowIndex,:]))
    stratreturngrossT = stratreturngross[rowIndex-1] +\
        np.sum(grossreturn[rowIndex,:])   
    
    # net
    ptfecT = ptfec[rowIndex-1]  * (1 + np.sum(netretT))
    ptfplT = ptfpl[rowIndex-1] + (100 * np.sum(netretT))
    stratreturnT = stratreturn[rowIndex-1] + np.sum(netretT)
    timeUnit_stratreturnT = stratreturnT - stratreturn[rowIndex-1]
    
    # -- Return output --
    return netretT, cumulnetretT, cumulgrossretT, ptfecgrossT, \
           ptfplgrossT, stratreturngrossT, ptfecT,\
           ptfplT, stratreturnT , timeUnit_stratreturnT
     
# -- Forces the column representation (I miss Matlab so much) -----------------
def rowNumpyArr_to_colNumpyArr(x):
    y = np.array(x)[:, np.newaxis]
    return y   

def makeNpArray_1Dimensional(x):
    y = np.array(x)[:,0]
    return y              
     
# -- Shift Backward -----------------------------------------------------------
# note: this forces the column representation

def ShiftBackward(x, mylag, method):
    try:
        [nrows,ncols] = x.shape     
    except:
        ncols = 1
        nrows = len(x)       
        x = rowNumpyArr_to_colNumpyArr(x)
    y = np.zeros((nrows,ncols))
    y[0:nrows-mylag,:] = x[mylag:nrows,:]  
    if method == 'carry over' or method == 'co':
        y[nrows-mylag:nrows,:] = np.tile(x[nrows-mylag,:],mylag)
    elif method == 'zero' or method == 'z':
        y=y
    return y           