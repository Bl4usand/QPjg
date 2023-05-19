# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:29:22 2020

@author: ASUS
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
from datetime import datetime#, timtedelta
import inspect
import modelUtilities as mutil


# -- Path manager --
mainDrive, masterDir, masterPath, projectDir, projecPath = \
    mutil.addPath('/home/dkuenzi/PycharmProjects/QPjg/',
     'QuantPlatformPy/','CRYPTO/NEWBRIDGE/')

# Create Quandl data object
class PortfolioPL():

    def __init__(self, dataStructure=None, signalInput=None, wgtInput=None, aum=None,\
                 assetClassTC=None, estimationCycle=None, tgtVolatility=None,\
                 nb=None, wgtGross=None, wgt=None, ExecP=None, grossreturn=None, tcforec=None,\
                 netret=None, cumulgrossret=None, ptfecgross=None, ptfplgross=None,\
                 stratreturngross=None, instGeoPlTemp=None, cumulnetret=None,\
                 ptfec=None, ptfpl=None, stratreturn=None, capital = None, \
                 capitalGeo=None, TC=None, anVolat=None, adjustedweight=None,\
                 maxLevj=None, adjFactorj=None, assetReturns=None, cumRet_usd=None):
        
        # -- Input --
        self.dataStructure = dataStructure
        self.aum = aum
        self.assetClassTC = assetClassTC
        self.wgtInput = wgtInput
        self.signalInput = signalInput

        # -- Instruments' wise --
        self.nb = nb
        self.wgtGross = wgtGross
        self.wgt = wgt
        self.ExecP = ExecP
        self.grossreturn = grossreturn
        self.tcforec = tcforec
        self.netret = netret
        self.cumulgrossret = cumulgrossret
        self.ptfecgross = ptfecgross
        self.ptfplgross = ptfplgross
        self.stratreturngross = stratreturngross
        self.instGeoPlTemp = instGeoPlTemp
        self.cumulnetret = cumulnetret
        self.TC = TC    
        self.adjustedweight = adjustedweight
        self.maxLevj = maxLevj
        self.adjFactorj= adjFactorj
        self.assetReturns = assetReturns           
        # -- Strategy's level --
        self.ptfec = ptfec
        self.ptfpl = ptfpl
        self.stratreturn = stratreturn
        self.cumRet_usd = cumRet_usd
        self.capital = capital
        self.capitalGeo = capitalGeo
        self.anVolat = anVolat
        self.tgtVolatility = tgtVolatility
        self.estimationCycle = estimationCycle
         
    #__________________________________________________________________
    # P&L
    def computePL(self):
        
        # -- Retrieve objects --
        aum = self.aum
        # Market data structure
        dataStructure = self.dataStructure
        # Market data dataFrames
        dfC = dataStructure.dfC
        dfO = dataStructure.dfO
        [nsteps, nc] = dfC.shape # Dimensions
        # Market data Numpy representation
        o = dfO.iloc[:,1:nc].to_numpy()
        c = dfC.iloc[:,1:nc].to_numpy() 
        ncols = nc-1
        
        # Signals, weights, ...
        signalInput     = self.signalInput
        wgtInput        = self.wgtInput
        assetClassTC    = self.assetClassTC 
        tgtVolatility   = self.tgtVolatility
        estimationCycle = self.estimationCycle
        
        # -- Pre-allocation -- 
        nsteps,  ncols,   nb, s, wgtGross, wgt, ExecP,  grossreturn,\
        tcforec, netret,  cumulgrossret,   ptfecgross,  ptfplgross,\
        stratreturngross, instGeoPlTemp,   cumulnetret, ptfec, ptfpl,\
        stratreturn,      capital,         capitalGeo,  TC, anVolat,\
        adjustedweight,   maxLevj,         adjFactorj,  assetReturns =\
        mutil.ptfOfFuturesLevel_PreallocationForPL(c, aum, assetClassTC)
        # -- Variables for volatility adjustment --
        inloop_returns = np.zeros([nsteps,1])
        inloop_std     = np.zeros([nsteps,1])
        scalingFactor  = np.ones([nsteps,1])
        finalWgt       = np.zeros([nsteps, ncols])
        
        # Dimension & transaction cost
        transactionCost = TC
    
        # Re-name signal an weights
        wgt = wgtInput.copy() #>=0 !!!!!!!!!!!!
        #s = np.sign(wgt)
        s = signalInput.copy()
        p  = mutil.ShiftBackward(o, 1, 'co') 
        #signedWeights = np.zeros([nsteps,ncols])
        
        for i in range(100,nsteps):
            
            # Strategy rolling-volatility
            inloop_std[i-1] = 16 * np.std(inloop_returns[i-65:i-1])
            # Scaling factor
            if inloop_std[i-1]==0 or np.isinf(inloop_std[i-1]) or np.isnan(inloop_std[i-1]):
                scalingFactor[i] = 1
            else:
                if estimationCycle[i] == 1:
                    scalingFactor[i] = min(tgtVolatility / inloop_std[i-1], 1)            
                elif estimationCycle[i] == 0:
                     scalingFactor[i] = scalingFactor[i-1]
            
            # -- INSTRUMENT LEVEL --
            for j in range(ncols):
                # -- Instrument-wise P&L --
                #finalWgt[i,j] = 1 * wgt[i,j]
                finalWgt[i,j] = scalingFactor[i] * wgt[i,j] #>=0
                [grossreturn_i,  tcforec_i] = mutil.Compute_StockFuture_PL(i, j,\
                 c, p, ExecP, s, finalWgt, nb, transactionCost)
                # -- Assign --
                grossreturn[i,j] = grossreturn_i;   
                tcforec[i,j] = tcforec_i;
            
            # -- SIGNED WEIGHTS --
            signedWeights = np.multiply(s, finalWgt) # ]-1;1[
            
            # -- PORTFOLIO LEVEL --  
            # -- Compute Portfolio's performance --
            netretT, cumulnetretT, cumulgrossretT, ptfecgrossT,  ptfplgrossT,\
                stratreturngrossT, ptfecT, ptfplT, stratreturnT, timeUnit_stratreturnT  = \
                    mutil.timeUnitPtfPerf(i, netret,   grossreturn, tcforec, \
                                            cumulnetret, ptfecgross,  ptfplgross, \
                                            stratreturn, stratreturngross, ptfec, ptfpl)                                                
                        
            # Assign
            netret[i,:] = netretT                # net return  
            instGeoPlTemp[i,:]  = np.multiply(instGeoPlTemp[i-1,:] , (np.ones((1,ncols)) + netret[i,:]))  # Update intermediaty P&L (geometric)
            cumulnetret[i,:]    = cumulnetretT   # cumulative net returns of instruments
            cumulgrossret[i,:]  = cumulgrossretT # cumulative gross returns of instruments
            ptfecgross[i]       = ptfecgrossT    # gross geometric curve
            ptfplgross[i]       = ptfplgrossT    # gross P&L
            stratreturngross[i] = stratreturngrossT # gross strategy's return
            ptfec[i]            = ptfecT         # net geometric curve
            ptfpl[i]            = ptfplT         # net P&L
            stratreturn[i]      = stratreturnT   # net strategy's return 
            inloop_returns[i]   = stratreturn[i] - stratreturn[i-1] # update return for rolling-volatility
        
        # Compute P&L
        cumRet_usd = stratreturn.copy()
        cumRet_usd = mutil.makeNpArray_1Dimensional(cumRet_usd)
        dailyRet_usd = mutil.Delta(cumRet_usd,'d',[1])
        
        # -- Return of the strategy --
        self.ptfecgross = ptfecgross
        self.ptfplgross = ptfplgross
        self.stratreturngross = stratreturngross
        self.ptfec = ptfec
        self.ptfpl = ptfpl
        self.stratreturn = stratreturn  
        self.dailyRet_usd = dailyRet_usd  
        self.capital = capital
        self.capitalGeo = capitalGeo
        self.anVolat = anVolat
        self.cumRet_usd = cumRet_usd
        # -- Weights per instrument --
        self.wgtGross = wgtGross
        self.wgt = wgt
        self.adjustedweight = adjustedweight
        self.maxLevj = maxLevj
        self.adjFactorj = adjFactorj        
        self.signedWeights = signedWeights
        # -- Returns per instrument -
        self.assetReturns = assetReturns
        self.ExecP = ExecP
        self.netret = netret
        self.instGeoPlTemp = instGeoPlTemp
        self.cumulnetret = cumulnetret
        self.cumulgrossret = cumulgrossret        
        self.grossreturn = grossreturn
        self.stratreturngross = stratreturngross
        self.scalingFactor = scalingFactor
        self.finalWgt = finalWgt
        self.inloop_std = inloop_std    