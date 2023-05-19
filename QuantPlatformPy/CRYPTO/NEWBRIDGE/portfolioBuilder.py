# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 00:45:18 2022

@author: user
"""
import pandas as pd  
import numpy as np
from numpy import inf
import riskfolio as rp

#______________________________________________________________________________
# ONE-TO-SIGMA PORTFOLIO

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
    # -- Return output ---               
    return z
#------------------------------------------------------------------------------ 

#------------------------------------------------------------------------------ 
# Rolling One-to-Sigma-based weights 
def weightsModel_One2Sigma(x, estimationCycle, lookback):
    # Dimensions
    (nsteps,ncols) = x.shape 
    # Compute rolling volatility
    retVol = 16*rollingVolatility(x,lookback)
    # Element-wise inverse
    retVolInv = np.power(retVol, -1)
    # inf to 0
    retVolInv[retVolInv == -inf] = 0
    retVolInv[retVolInv == inf] = 0
    # Sum
    sumRetVolInv = np.sum(retVolInv, axis=1)  
    sumRetVolInv = sumRetVolInv.reshape(-1,1)
    sumRetVolInv = np.tile(sumRetVolInv,(1,ncols))
    # FInal weights
    weigthVolInv = np.divide(retVolInv, sumRetVolInv)
    weigthVolInv = np.nan_to_num(weigthVolInv)
    # Adjust weights only when the portoflio is reshufled
    weigthVolInv_cycle = weigthVolInv
    for i in range(1,nsteps):
        if estimationCycle[i]==0:
            weigthVolInv_cycle[i,:]=weigthVolInv_cycle[i-1,:]
    # -- Return output --
    return weigthVolInv_cycle                
#------------------------------------------------------------------------------

#______________________________________________________________________________
# ROLLING HIERARCHICAL RISK PARITY

#-- Parameters for HRP --------------------------------------------------------
model        = 'HRP'     # Could be HRP or HERC
codependence = 'pearson' # Correlation matrix used to group assets in clusters
rm           = 'MV'      # Risk measure used, this time will be variance
rf           = 0         # Risk free rate
linkage      = 'single'  # Linkage method used to build clusters
#max_k        = 10        # Max number of clusters used in two difference gap statistic, only for HERC model
leaf_order   = True      # Consider optimal order of leafs in dendrogram

#-- Rolling HRP ---------------------------------------------------------------
def rolling_HRP(x, dateName, estimationCycle, periodForReturns, lookback):
    
    # note: x is a data frame with dates

    # x is data frame index/date/data : drop the date
    x = x.drop(dateName, 1)
    # Dimensions & pre-allocation
    (nsteps, ncols) = x.shape     
    #columnsIdx = np.arange(ncols) # Create a row vector of colIdx    
    
    # Returns
    returns = x.pct_change(periods=periodForReturns, fill_method='pad')
    returns = returns.fillna(0)
           
    # Pre-allocation
    optimal_weights = np.zeros((nsteps,ncols))
    
    # Convert dataframe to numpy (this is a price)
    xnpa = x.to_numpy(dtype=None)
    
    # Main-loop
    # note: I start at "2*lookback" because i need to be sure that the returns
    #       extracted from time="i-lookback" to time="i" are for instruments 
    #       which have been trading over the considered period. 
    for i in range(2*lookback+1, nsteps):
        
        if estimationCycle[i]==1:
            
            # In order to be able to compute the covariance matrix, one needs
            # to observe the returns over a period of "lookback" bar. 
            # Therefore, instrument[j] must have started trading "lookback" bars ago 
            # (xnpa is a close price).
            # Find non-Zero
            solution = 2
            if solution == 1:
                xnpa_check =  xnpa[i-lookback,:]  # I take the first row only             
                idxNon0 = list(xnpa_check.nonzero())# problem with format
            elif solution == 2:
                xnpa_check =  xnpa[i-lookback,:]  # I take the first row only               
                idxNon0 = [ u for u in range(ncols) if xnpa_check[u] != 0 ]
            elif solution == 3:   
                xnpa_check =  xnpa[i-lookback:i,:] # U take the full matrix                 
                # First row-wise sum (if ==0, then no price) 
                sum_rowWise = np.sum(xnpa_check, axis=0)
                # Identify which is one is non-zero
                idxNon0 = [ u for u in range(ncols) if sum_rowWise[u] != 0 ]
   
            # Extract & clean matrix of returns for non zeros      
            returns_e =  returns.iloc[i-lookback:i, idxNon0]
            returns_e = returns_e.replace(np.nan, 0)
            returns_e = returns_e.replace(np.inf, 0)
            returns_e = returns_e.replace(-np.inf, 0)
                                    
            # Estimate mean returns & covariance matrix
            #mean_returns = returns_e.mean() # not used  
            #cov_matrix_snap = returns_e.cov()             
            
            # Max nb of clusters used in 2 difference gap statistic for HERC model
            # bote: 3 instruments minimum per cluster
            minNbofCoinsPerCluser = 3
            max_k = np.floor(returns_e.shape[1] / minNbofCoinsPerCluser)       
                        
            # Build the HRP-portfolio object
            port = rp.HCPortfolio(returns = returns_e)         
            
            # Compute HRP-based weights for the subset of extracted instruments
            w_hrp = port.optimization(model=model,
                                  codependence=codependence,
                                  rm=rm,
                                  rf=rf,
                                  linkage=linkage,
                                  max_k=max_k,
                                  leaf_order=leaf_order)    
            optimal_weights_snap = w_hrp.to_numpy()     
            optimal_weights_snap = np.array(optimal_weights_snap)[:,0]
            optimal_weights_snap = np.nan_to_num(optimal_weights_snap)  
            
            # Now re-map ERC-based weights of the subset of extracted 
            # instruments into the weights matrix of dim ncols >= len(idxNon0)
            for u in range(len(idxNon0)):
                optimal_weights[i, idxNon0[u]] = optimal_weights_snap[u]
            
        elif estimationCycle[i]==0:   
            
            optimal_weights[i,:] = optimal_weights[i-1,:]
            
    # -- Return output --
    return optimal_weights 



