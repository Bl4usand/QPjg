# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 18:20:08 2022

@author: user
"""

#______________________________________________________________________________
# IMPORT PACKAGES, DEFINE PATHS, IMPORT API-KEYS

import pandas as pd
import numpy as np
import modelUtilities as mutil
from cryptoDataClass import dataStructureCsv, dataStructureCryptoCompare

# -- Add paths & Manage dependencies --
mainDrive, masterDir, masterPath, projectDir, projectPath = mutil.addPath('/home/dkuenzi/PycharmProjects/QPjg/',
     'QuantPlatformPy/','CRYPTO/NEWBRIDGE/')

# mantime, npu, dfu, linch, stfun, volptf, eqsptf,\
#     futbu, futptbu, mvow, ercw, hrpw = mutil.manage_model_dependencies(masterPath)

def download_HistoricalData(downloadDataOption):

    '''downloadDataOption = "donwload CC data"
                          = "donwload CC data & concatenateCSV" 
                          = "fetch CSV data" '''

    # -- Define universe & meta-data --
    exchange, timeframe, nbOfObs, ccy1, ccy2, instTC, aum, budgets, \
        tgtVolatility, outputDir, histDataDir, dataDir_fullUniverse,\
            histDataDir_fullUniverse = mutil.modelConfiguration("fullUniverse", masterDir, projectDir)
            
    # Re-set nbOfObs      
    if downloadDataOption == "donwload CC data & concatenateCSV":
        nbOfObs=15

    # -- Choose download option --  
    if downloadDataOption == "fetch CSV data":  
        # -- retrieve --
        dataStructure = dataStructureCsv(histDataDir_fullUniverse)
        dataStructure.getCsvHistoricalData(histDataDir_fullUniverse)
        
    elif downloadDataOption == "donwload CC data":
        # -- download --
        dataStructure = dataStructureCryptoCompare(ccy1, ccy2, nbOfObs, timeframe, histDataDir_fullUniverse)
        dataStructure.downloadCCdata()
        # -- write --
        # Open price
        csvName = 'openPrice.csv'
        csvPath = dataDir_fullUniverse + csvName
        dataStructure.dfO.to_csv(csvPath, index=False)
        # High price
        csvName = 'highPrice.csv'
        csvPath = dataDir_fullUniverse + csvName
        dataStructure.dfH.to_csv(csvPath, index=False)    
        # Low price
        csvName = 'lowPrice.csv'
        csvPath = dataDir_fullUniverse + csvName
        dataStructure.dfL.to_csv(csvPath, index=False)
        # Close price
        csvName = 'closePrice.csv'
        csvPath = dataDir_fullUniverse + csvName
        dataStructure.dfC.to_csv(csvPath, index=False)   
        # Volume to
        csvName = 'volumeTo.csv'
        csvPath = dataDir_fullUniverse + csvName
        dataStructure.dfVto.to_csv(csvPath, index=False)   
        # Volume from
        csvName = 'volumeFrom.csv'
        csvPath = dataDir_fullUniverse + csvName
        dataStructure.dfVfrom.to_csv(csvPath, index=False) 
        
    elif downloadDataOption == "donwload CC data & concatenateCSV":
        # -- download --        
        dataStructure = dataStructureCryptoCompare(ccy1, ccy2, nbOfObs, timeframe, histDataDir_fullUniverse)
        dataStructure.downloadCCdata_ConcatHistCSV(histDataDir_fullUniverse)
        # -- write --
        # Open price
        csvName = 'openPrice.csv'
        csvPath = histDataDir_fullUniverse + csvName
        dataStructure.dfO.to_csv(csvPath, index=False)
        # High price
        csvName = 'highPrice.csv'
        csvPath = histDataDir_fullUniverse + csvName
        dataStructure.dfH.to_csv(csvPath, index=False)    
        # Low price
        csvName = 'lowPrice.csv'
        csvPath = histDataDir_fullUniverse + csvName
        dataStructure.dfL.to_csv(csvPath, index=False)
        # Close price
        csvName = 'closePrice.csv'
        csvPath = histDataDir_fullUniverse + csvName
        dataStructure.dfC.to_csv(csvPath, index=False)   
        # Volume to
        csvName = 'volumeTo.csv'
        csvPath = histDataDir_fullUniverse + csvName
        dataStructure.dfVto.to_csv(csvPath, index=False)   
        # Volume from
        csvName = 'volumeFrom.csv'
        csvPath = histDataDir_fullUniverse + csvName
        dataStructure.dfVfrom.to_csv(csvPath, index=False)         
