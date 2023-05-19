# -*- coding: utf-8 -*-
"""
COMPUTE FACTORS FOR CRYPROS
"""

# -- Import packages --
#import importlib, importlib.machinery
#import sys
#import setuptools, imp, csv
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
from numpy import inf
#import datetime as dt  
#from datetime import datetime#, timtedelta
#import matplotlib.pyplot as plt
#import seaborn as sns
# import cryptocompare as cryc # NO NEED!!!! request the url ofthe APi !!!!!!
#import requests
#import inspect
#from DataClass_Cryptocompare import CryptocompData
#from FactorClass_Crypto import FactorsData
#import modelUtilities as mutil
import modelUtilities as mutil
import pyrb
from pykalman import KalmanFilter  # For Kalman Filter
import statsmodels.api as sm       # For OLS

# -- Path manager --
#mainDrive, masterDir, masterPath, projectDir, projecPath = mutil.addPath('C:/',
#     'QuantPlatformPy/','DigitalPreciousMoney/')
mainDrive, masterDir, masterPath, projectDir, projectPath = mutil.addPath('/home/dkuenzi/PycharmProjects/QPjg/',
     'QuantPlatformPy/','CRYPTO/NEWBRIDGE/')

# -- My packages --
# sys.path.insert(0,masterPath+'statisticalFunctions') 
# stfun = importlib.import_module('statFunctions', __name__)
# sys.path.insert(0,masterPath+'technicalIndicators')
# teki = importlib.import_module('technicalIndicators_functions', __name__)


#______________________________________________________________________________
class FactorsData():
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, nameList=None, dataStructure=None):
        """
        .........
        """
        # Kalman Filtered data
        self.nameList = nameList
        self.dataStructure = dataStructure
        #self.h = h
        #self.l = l
        #self.dfC = dfC
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # -- KALMAN FILTERS --
    def KF_NonConstCoeff(self):
        dataStructure = self.dataStructure
        dataDf = dataStructure.dfC
        # Extract nameList
        nameList = []
        for col in dataDf.columns:
            nameList.append(col)
        del nameList[0] 
        [nr,nc] = dataDf.shape
        kfx = np.zeros((nr,1))
        kfx = kfx.reshape(-1,1)
        for j in range(1,nc):
            # KF initialisation (non-optimzed - improve this !!!)
            kfmodel = KalmanFilter(transition_matrices = [1], \
                                   observation_matrices = [1],initial_state_mean = 0,\
                                   initial_state_covariance = 1, \
                                   observation_covariance=1,transition_covariance=.01)
            # Use the observed values of the price to get a rolling mean
            dataDf_j = dataDf.iloc[:,j]
            dataDf_j = dataDf_j.to_numpy()
            j_SateMeans, _ = kfmodel.filter(dataDf_j)
            #j_SateMeans_df = pd.Series(j_SateMeans.flatten(),dfC.index)
            #dTemp_df = pd.Series(dfC[:,1],dfC.index)
            dTemp = j_SateMeans.reshape(-1,1)
            kfx = np.append(kfx, dTemp, axis=1)
        # Delete first pre-allocation column    
        kfx = np.delete(kfx,0,1)
        # Dataframe
        #kfxdf = pd.DataFrame(data=kfx, index=dataDf["time"], columns=nameList)
        kfxdf = pd.DataFrame(data=kfx, index=dataDf.iloc[:,0], columns=nameList)
        kfxdf = kfxdf.reset_index()
        self.kfx = kfx
        self.kfxdf = kfxdf
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SET OF TECHNICAL FACTORS  
    def Compute_Technicals(self):    
        
        # -- Retrieve O,H,L,C & Volume --
        dataStructure = self.dataStructure
        dfH = dataStructure.dfH
        dfL = dataStructure.dfL
        dfC = dataStructure.dfC
        # Dimensions
        [nn,nc] = dfC.shape
        # Numpy representation for teki
        h = dfH.iloc[:,1:nc].to_numpy()
        l = dfL.iloc[:,1:nc].to_numpy()
        c = dfC.iloc[:,1:nc].to_numpy()
        
        # -- Momenta --
        dayRet = mutil.Delta(c, 'roc', [1])
        self.dayRet = dayRet
        
        # # -- Stop-And-Reverse --
        # sar = teki.stopAndReverse(h, l, 0.04, 0.2)
        # self.sar = sar
        
        # # -- Savitzky-Golay --
        # # lookbak @ 21 bars - polynomial order: 1, 2, 3, 4
        # lookback = 21
        # z211 = teki.rolling_Savitzky_Golay_filter(c,[lookback,1]) 
        # z212 = teki.rolling_Savitzky_Golay_filter(c,[lookback,2]) 
        # z213 = teki.rolling_Savitzky_Golay_filter(c,[lookback,3]) 
        # z214 = teki.rolling_Savitzky_Golay_filter(c,[lookback,4])          
        # self.z211 = z211
        # self.z212 = z212
        # self.z213 = z213
        # self.z214 = z214        
        # # lookbak @ 25 bars - polynomial order: 1, 2, 3, 4
        # lookback = 25
        # z251 = teki.rolling_Savitzky_Golay_filter(c,[lookback,1])
        # z252 = teki.rolling_Savitzky_Golay_filter(c,[lookback,2])
        # z253 = teki.rolling_Savitzky_Golay_filter(c,[lookback,3])
        # z254 = teki.rolling_Savitzky_Golay_filter(c,[lookback,4])
        # self.z251 = z251
        # self.z252 = z252
        # self.z253 = z253
        # self.z254 = z254  
        # # lookbak @ 125 bars - polynomial order: 1, 2, 3, 4
        # lookback = 125
        # z1251 = teki.rolling_Savitzky_Golay_filter(c,[lookback,1])
        # z1252 = teki.rolling_Savitzky_Golay_filter(c,[lookback,2])
        # z1253 = teki.rolling_Savitzky_Golay_filter(c,[lookback,3])
        # z1254 = teki.rolling_Savitzky_Golay_filter(c,[lookback,4])  
        # self.z1251 = z1251
        # self.z1252 = z1252
        # self.z1253 = z1253
        # self.z1254 = z1254        
        # # lookbak @ 55 bars - polynomial order: 1, 2, 3, 4
        # lookback = 55
        # z551 = teki.rolling_Savitzky_Golay_filter(c,[lookback,1]) 
        # z552 = teki.rolling_Savitzky_Golay_filter(c,[lookback,2]) 
        # z553 = teki.rolling_Savitzky_Golay_filter(c,[lookback,3]) 
        # z554 = teki.rolling_Savitzky_Golay_filter(c,[lookback,4])          
        # self.z551 = z551
        # self.z552 = z552
        # self.z553 = z553
        # self.z554 = z554
        # # lookbak @ 89 bars - polynomial order: 1, 2, 3, 4
        # lookback = 89
        # z891 = teki.rolling_Savitzky_Golay_filter(c,[lookback,1]) 
        # z892 = teki.rolling_Savitzky_Golay_filter(c,[lookback,2]) 
        # z893 = teki.rolling_Savitzky_Golay_filter(c,[lookback,3]) 
        # z894 = teki.rolling_Savitzky_Golay_filter(c,[lookback,4])          
        # self.z891 = z891
        # self.z892 = z892
        # self.z893 = z893
        # self.z894 = z894
        
        # # -- Relative Stength Index --
        # rsi3 = teki.rsi(c,3)
        # self.rsi3 = rsi3
        # rsi14 = teki.rsi(c,14)
        # self.rsi14 = rsi14        
        
        # -- Regressions slope --
        def rollingOLS(c, returnLookback, estimationLookback):
            instReturns = mutil.Delta(c,'roc', [returnLookback])
            [nr,nc] = c.shape
            ols_beta = np.zeros((nr,nc))
            ols_alpha = np.zeros((nr,nc))
            ols_rsquared = np.zeros((nr,nc))
            lookback = estimationLookback
            timeVector =  np.arange(1,lookback+1)
            #rsquared_in = 1;
            for j in range(nc):
                retj = instReturns[:,j]
                for i in range(lookback, nr):
                    try:
                        #retji = retj[i-lookback:i]
                        retji = retj[i-lookback+1:i+1] # This is correct (only for column vector?)
                        X = timeVector
                        X = sm.add_constant(X) # No constant is added by the model unless you are using formulas
                        olsmod = sm.OLS(retji, X)
                        results = olsmod.fit()
                        olsmod_params = results.params
                        olsmod_tvalues = results.tvalues  # print(olsmod_tvalues([1, 0]))
                        olsmod_summary = results.summary  # print(olsmod_summary())
                        rsquared = results.rsquared
                        ols_alpha[i,j] = olsmod_params[0]# * rsquared**rsquared_in
                        ols_beta[i,j] =  olsmod_params[1]# * rsquared**rsquared_in
                        ols_rsquared[i,j] = rsquared
                    except:
                        ols_alpha[i,j] = ols_alpha[i-1,j]
                        ols_beta[i,j] = ols_beta[i-1,j]
                        ols_rsquared[i,j] = ols_rsquared[i-1,j]
            return ols_alpha, ols_beta, ols_rsquared
        
        ols_alpha1, ols_beta1, ols_rsquared1 = rollingOLS(c, 21, 21)
        ols_alpha2, ols_beta2, ols_rsquared2 = rollingOLS(c, 21, 34)
        ols_alpha3, ols_beta3, ols_rsquared3 = rollingOLS(c, 10, 21)
        ols_alpha4, ols_beta4, ols_rsquared4 = rollingOLS(c, 10, 34)        
        
        ols_rsquared = ols_rsquared1 + ols_rsquared2 + ols_rsquared3 + ols_rsquared4
        
        ols_alpha = np.multiply(np.divide(ols_rsquared1, ols_rsquared), ols_alpha1) + \
                    np.multiply(np.divide(ols_rsquared2, ols_rsquared), ols_alpha2) + \
                    np.multiply(np.divide(ols_rsquared3, ols_rsquared), ols_alpha3) + \
                    np.multiply(np.divide(ols_rsquared4, ols_rsquared), ols_alpha4)
        ols_alpha = np.nan_to_num(ols_alpha)   

        ols_beta = np.multiply(np.divide(ols_rsquared1, ols_rsquared), ols_beta1) + \
                   np.multiply(np.divide(ols_rsquared2, ols_rsquared), ols_beta2) + \
                   np.multiply(np.divide(ols_rsquared3, ols_rsquared), ols_beta3) + \
                   np.multiply(np.divide(ols_rsquared4, ols_rsquared), ols_beta4)        
        ols_beta = np.nan_to_num(ols_beta) 
        
        self.ols_beta = ols_beta        
        self.ols_alpha = ols_alpha  
        