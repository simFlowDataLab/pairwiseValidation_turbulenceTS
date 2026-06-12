#
import sys
import numpy as np
import pickle

def channelTS_read(dataPath_):
    """
    Read time series samples of turbulent channel flow simulation

    Args: 
       `dataPath_`: string, path to the channel flow time series data

    Return:
       `t`: numpy array of size n, timestamps
       `y`; numpy array of size ny, wall-distance of the sampling points (probes)
       `ypls`: numpy array of size ny, wall-distance of the sampling points (probes) in wall units
       `u`: numpy array of shape ny x n, samples of streamwise velocity at ny probes
       `uTau`: numpy array of size n, samples of the wall friction velocity
    """
    #Constants
    nu=1./5019.22   #kinematic viscosity
    Ub=1.0
    delta=1.0
    uTau0=0.0594    #reference uTau

    #time samples
    with open(dataPath_+'t', 'rb') as f:
         t = pickle.load(f)

    #samples of streamwise velocity averaged over wall-parallel planes
    with open(dataPath_+'u', 'rb') as f:
         u = pickle.load(f)

    # Wall-distance of the sampling points (normalized by channel half-height)
    with open(dataPath_+'y', 'rb') as f:
         y = pickle.load(f)

    # samples of wall friction velocity
    with open(dataPath_+'uTau', 'rb') as f:
         uTau = pickle.load(f)

    ypls=y*298

    return t,y,ypls,uTau,u

def channelDataSelect(iLocList,u,iDiscard=0,iSkip=10):
    """
    Selection of time series data at a subest of locations available in the database
    
    Args:
       `iLocList`: list of integers - (indices associated with wall-normal coordinates)
                   length of iLocList = K = number of variables/variates
                   Note: minimum K = 2 (two time series)
       `u`: numpy array of shape nyxn, n samples of streamwise velocity at ny probes
       `iDiscard`, int, number of initial samples to discard due to transition, if any (if not, =0)
       `iSkip`, int>0, sampling frequency 
       
    Returns:
       `fSim`: umpy array of shape (number of samples x number of variates)
    """
    if len(iLocList)<2:
       raise ValueError("At least two locations should be provided.") 

    I = iLocList
    u_ = u[iDiscard:,I]
    fSim = u_[::iSkip,:]    
        
    return fSim
