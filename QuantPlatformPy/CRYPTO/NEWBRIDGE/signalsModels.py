# -*- coding: utf-8 -*-
"""
EXTRACT SIGNALS
"""


# -- Import packages --
import importlib, importlib.machinery
import sys
#import setuptools, imp, csv
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
from numpy import inf
import datetime as dt  
from datetime import datetime #, timtedelta
import inspect
import modelUtilities as mutil
import pykalman as pyk
import talib as tal
import statsmodels.api as sm  

# -- Path manager --
mainDrive, masterDir, masterPath, projectDir, projecPath =\
    mutil.addPath('/home/dkuenzi/PycharmProjects/QPjg/',
     'QuantPlatformPy/','CRYPTO/NEWBRIDGE/')


# -- Utilities functions ------------------------------------------------------
# note: this forces the column representation
def rowNumpyArr_to_colNumpyArr(x):
    y = np.array(x)[:, np.newaxis]
    return y   

def makeNpArray_1Dimensional(x):
    y = np.array(x)[:,0]
    return y   

# -- Relative Strength Index --------------------------------------------------
# example: y = rsi(c,14)
def _rsi(c,lookbackPeriod):
    # Dimension
    try:
        (nsteps,ncols) = c.shape 
    except:
        nsteps = len(c)
        ncols=1        
    # Preallocation
    y = np.zeros((nsteps,1))
    # Main loop
    for j in range(ncols):
        # Compute
        ytemp = tal.RSI(c[:,j], timeperiod = lookbackPeriod)
        # Append
        ytemp = rowNumpyArr_to_colNumpyArr(ytemp)
        y     = np.append(y, ytemp, axis=1)
    # Clean data
    y = np.delete(y,0,1)# delete first pre-allocation column    
    y = np.nan_to_num(y)
    # return value
    return y

# -- Stop & Reverse -----------------------------------------------------------  
# example: tki.stopAndReverse(h,l, 0.04, 0.2)
def _stopAndReverse(h,l, acceleration, maxStep):
    # Dimension
    (nsteps,ncols) = l.shape 
    # Preallocation
    y = np.zeros((nsteps,1))
    # Main loop
    for j in range(ncols):
        # Compute
        ytemp = tal.SAR(h[:,j], l[:,j], acceleration = acceleration, maximum = maxStep)  
        # Append
        ytemp = rowNumpyArr_to_colNumpyArr(ytemp)
        y = np.append(y, ytemp, axis=1)
    # Clean
    y = np.delete(y,0,1)# delete first pre-allocation column    
    y = np.nan_to_num(y)
    
    # return value
    return y

# -- Weighted returns & P&L --
def TrendFollowgingModel(dataStructure, factorsSet, weightsInput, modelOption):
    
    # Market data dataFrames
    dfO = dataStructure.dfO
    dfH = dataStructure.dfH
    dfL = dataStructure.dfL
    dfC = dataStructure.dfC
    dfVfrom = dataStructure.dfVfrom
    dfVto = dataStructure.dfVto
    [nsteps,nc] = dfC.shape # Dimensions
    # Market data Numpy representation
    o = dfO.iloc[:,1:nc].to_numpy()
    h = dfH.iloc[:,1:nc].to_numpy()
    l = dfL.iloc[:,1:nc].to_numpy()
    c = dfC.iloc[:,1:nc].to_numpy() 
    vfrom = dfVfrom.iloc[:,1:nc].to_numpy() 
    vto = dfVto.iloc[:,1:nc].to_numpy()      
    # Kalman & other technical factors
    kfx = factorsSet.kfx
    
    # -- Dimensions & Preallocation --
    ncols = nc-1
    s = np.zeros((nsteps,ncols))
    
    if modelOption == 'always in':   
        
        modelWeights = weightsInput.copy()
        s = np.ones((nsteps,ncols))
        
    if modelOption == 'multiKernels2':
        
        modelWeights = weightsInput.copy()
        
        # - Price-to-Kalman-distance - 
        dist2Kalman = c - kfx
        
        # - RSI momentum -
        rsi  = _rsi(c, 14)
        #rsimom = rsi - teki.ema(rsi, 21)
        
        # - Parabolic / Stop-and-Reverse
        sar = _stopAndReverse(h,l, 0.05, 0.3)
        
        # - Trend index -
        trendIndex  = np.sign(
                              np.sign(dist2Kalman) +\
                              1*np.sign(factorsSet.ols_alpha) +\
                              1*np.sign(factorsSet.ols_beta)
                              )     
            
        # - Money management -
        entryLevel  = np.zeros([nsteps, ncols])
        maxLongIncursion = np.zeros([nsteps, ncols])
        londDuration = np.zeros([nsteps, ncols])
        modelWeights_memo = modelWeights.copy()
        weigthAdjustment = np.zeros([nsteps, ncols])
        
        # - Multi-break out -
        maxLookback = 100
        for i in range(maxLookback,nsteps):
            for j in range(0,ncols):
                # Fully invested
                #my_string = "Row {ri} - trendIndex has value {ti}.".format(ri=i, ti=trendIndex[i,j])
                #print(my_string)
                if s[i-1, j] != 1 and trendIndex[i,j] > 0 and vto[i,j] > vto[i-5,j]: 
                #if s[i-1, j] != 1 and rsi3[i,j]>0:     
                    #if c[i,j] > sar[i,j]:
                    modelWeights[i,j]     = 1 * modelWeights_memo[i,j]   
                    weigthAdjustment[i,j]  = 1
                    s[i,j]                = 1 
                    entryLevel[i,j]       = c[i,j]
                    maxLongIncursion[i,j] = c[i,j]
                    londDuration[i,j]     = 1 
                    #else:
                    #    modelWeights[i,j] = 1 * modelWeights[i,j]   
                    #    s[i,j] = 1    
                elif s[i-1, j] == 1 and trendIndex[i,j] > 0 and rsi[i,j]<90:      
                    modelWeights[i,j]     = 1 * modelWeights_memo[i,j]  
                    weigthAdjustment[i,j] = 1
                    entryLevel[i,j]       = entryLevel[i-1,j]
                    maxLongIncursion[i,j] = max(maxLongIncursion[i-1,j], c[i,j])
                    londDuration[i,j]     = londDuration[i-1,j]+1
                    s[i,j] = 1                     
                # Trend exhaustion (reduce)
                elif s[i-1, j] != 1 and trendIndex[i,j] < 0 \
                    and (kfx[i,j] > kfx[i-5,j] and c[i,j]>sar[i,j]): 
                    modelWeights[i,j] = 0.5 * modelWeights_memo[i,j]   
                    weigthAdjustment[i,j]  = 0.5
                    s[i,j] = 1    
                    entryLevel[i,j]       = c[i,j]
                    maxLongIncursion[i,j] = c[i,j]
                    londDuration[i,j]     = 1   
                # Trend exhaustion (reduce)
                elif s[i-1, j] == 1 and trendIndex[i,j] < 0 \
                    and c[i,j] > sar[i,j] and rsi[i,j]<90: 
                    modelWeights[i,j] = 0.5 * modelWeights_memo[i,j]   
                    weigthAdjustment[i,j]  = 0.5
                    maxLongIncursion[i,j] = max(maxLongIncursion[i-1,j], c[i,j])
                    londDuration[i,j]     = londDuration[i-1,j]+1
                    s[i,j] = 1 
    # -- Return output --
    return s, modelWeights, kfx, c, weigthAdjustment
