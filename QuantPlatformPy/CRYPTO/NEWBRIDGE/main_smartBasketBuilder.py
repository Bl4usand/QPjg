
#______________________________________________________________________________
# IMPORT PACKAGES, DEFINE PATHS, IMPORT API-KEYS

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import modelUtilities as mutil
from cryptoDataClass import dataStructureCsv_fromFullUniverse
from cryptoFactorClass import FactorsData
import signalsModels as simo
from portfolioClass import PortfolioPL
import download_HistDataFullUniverse as dhfu
import portfolioBuilder as ptfb

# -- Add paths & Manage dependencies --
mainDrive, masterDir, masterPath, projectDir, projectPath = mutil.addPath('/home/dkuenzi/PycharmProjects/QPjg/',
     'QuantPlatformPy/','CRYPTO/NEWBRIDGE/')

#______________________________________________________________________________
# DOWNLOAD HISTORICAL DATA FULL UNIVERSE

# note: the first time the code was run, downloadDataOption="donwload CC data"
#       then downloadDataOption="donwload CC data & concatenateCSV" 
downloadFullUni=True
if downloadFullUni:
    dhfu.download_HistoricalData("donwload CC data & concatenateCSV")

#______________________________________________________________________________
# EXTRACT SUB-UNIVERSE DATA FROM FULL UNIVERSE

# -- Define universe & meta-data --
exchange, timeframe, nbOfObs, ccy1, ccy2, instTC, aum, budgets, \
    tgtVolatility, outputDir, histDataDir, dataDir_fullUniverse,\
        histDataDir_fullUniverse = mutil.modelConfiguration("coreUniverse", masterDir, projectDir)
# -- Choose download option --  
#downloadDataOption  = ["donwload CC data", "donwload CC data & concatenateCSV", 
#                       "fetch CSV data"]
#downloadDataOption = "donwload CC data & concatenateCSV"
getDataFromCsv = True
dataStructure = dataStructureCsv_fromFullUniverse(ccy1, histDataDir_fullUniverse)
dataStructure.getCsvHistoricalData(ccy1, histDataDir_fullUniverse)

#csvPath = 'C:/QuantPlatformPy/CRYPTOS/output_data/' + 'dfC2RT.csv'
#dataStructure.dfC.to_csv(csvPath)  

#______________________________________________________________________________
# COMPUTE FACTORS

computeFactors = True
if computeFactors:
    factorsSet = FactorsData(ccy1, dataStructure) #Instantiate the class
    # -- Compute factors for extracting signals --
    factorsSet.KF_NonConstCoeff()    
    factorsSet.Compute_Technicals()
    
#csvPath = 'C:/QuantPlatformPy/CRYPTOS/output_data/' + 'kfxdf2RT.csv'
#factorsSet.kfxdf.to_csv(csvPath)      
    
#______________________________________________________________________________
# COMPUTE WEIGHTS

# -- Define cycle for weight recalibration --
# note: weights are kept fixed within a given cycle
estimationCycle, dateArray, estimationCycleTot = mutil.calculationCycle(dataStructure.benchdate_df, "series", 'm') 

# -- Weights for rolling optimization --   
portoflioConstructor = 'hrp' #{'one2sigma', 'hrp'}
# One-to-Sigma weights
if portoflioConstructor == 'one2sigma':
    weigthVolInv = ptfb.weightsModel_One2Sigma(factorsSet.dayRet, estimationCycle, 34)  
    chosenWeights = weigthVolInv.copy()
# Hierarchical Risk Parity structure
elif portoflioConstructor == 'hrp':
    weightHrp = ptfb.rolling_HRP(dataStructure.dfC, 'time', estimationCycle, 1, 34)   
    chosenWeights = weightHrp.copy()
# -- Write these weights before signal * weights for memo
pureWeightsDf = pd.DataFrame(data=chosenWeights, index=dataStructure.benchdate_np, columns=ccy1)
#pureWeightsDf.to_csv(outputDir+'pureWeights.csv', index=False)

#______________________________________________________________________________
# SIGNALS & SIGNED WEIGHTS  

s, modelWeights, kfx, c, weigthAdjustment = \
    simo.TrendFollowgingModel(dataStructure, factorsSet, chosenWeights, 'multiKernels2')
  
#_____________________________________________________________________________
# COMPUTE P&L

# -- Instantiate the class --                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      t                                                
ptfplClass = PortfolioPL(dataStructure)
# -- Assign instrument-level data & parameters --
ptfplClass.signalInput = s
ptfplClass.wgtInput = modelWeights
ptfplClass.aum = aum
ptfplClass.assetClassTC = instTC
ptfplClass.estimationCycle = estimationCycle
ptfplClass.tgtVolatility = tgtVolatility
# -- Compute P&L --
ptfplClass.computePL()    
# retrieve some useful objects for checking
scalingFactor = ptfplClass.scalingFactor # scaling factor
inloop_std = ptfplClass.inloop_std       # running rolling-volatility

#_____________________________________________________________________________
# CARRY BUCKET

# -- Final scaled weights --
finalwgt = ptfplClass.wgt               
sumWgt = finalwgt.sum(axis=1)
## -- Carry bucket --
# Percentage of the book to lending
allocToCarry = np.ones(dataStructure.dfC.shape[0]) - sumWgt
#plt.plot(allocToCarry); plt.show()
# Carry time series (set to 0 for now)
lendingProtocol_yield = 0/100;
dailyPl_carry = allocToCarry * lendingProtocol_yield  / 365
pl_Carry = np.cumsum(dailyPl_carry)
#plt.plot(pl_Carry ); plt.show()

#______________________________________________________________________________
# FINAL P&L & RISK METRICS

# -- Arithmetic representation --
finaPL = pl_Carry + ptfplClass.cumRet_usd
plDf = pd.DataFrame({'timestamp':dataStructure.benchdate_np, 'pl': finaPL})
plDf['timestamp'] = plDf['timestamp'].dt.date       
mutil.plotWrapper_Format1(plDf,'timestamp','pl','blue','Strategy return','option2')
tsdd, maxdd, time_maxdd = mutil.computeDrawdownTS(finaPL, 'arithmetic')

# -- Combine with BTC --
# note: I transform BTC price, i.e. a geometric compounded quantity by definition
#       into an arithmetic compounded quantity and then adjust it by relarive volatility
# Find first non-zero value in the numpy array
sr = (finaPL!=0).argmax(axis=0)
# Extract BTC & Compute return
btc = dataStructure.dfC['BTC'].to_numpy()
btcRet = mutil.Delta(btc,'roc',[1])
btcRet[:sr] = 0
# Adjust btcRet by relative volatility
btcSigma = np.std(btcRet[sr:])
finaPLRet = mutil.Delta(finaPL,'d',[1])
finaPLSigma = np.std(finaPLRet[sr:])
adjBtcRet = btcRet * finaPLSigma/btcSigma
adjBtcCumsumArithRet = np.cumsum(adjBtcRet)
xDf = pd.DataFrame({'Date':dataStructure.benchdate_np[sr-1:],\
                  'Price': finaPL[sr-1:], 'Btc':adjBtcCumsumArithRet[sr-1:]})  

#______________________________________________________________________________
# WRITE OUTPUT TO CSV

# mutil.write_outputCsv("fetch CSV data", outputDir, histDataDir, \
#    dataStructure, plDf, ptfplClass, dataStructure.benchdate_np, ccy1)
 
if __name__ == "__main__":
    print("Run Smart basket")
    pass
