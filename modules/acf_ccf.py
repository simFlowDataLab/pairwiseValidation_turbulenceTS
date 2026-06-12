#
# https://github.com/simFlowDataLab/PairwiseValidation_turbulenceTS
# Saleh Rezaeiravesh, saleh.rezaeiravesh@manchester.ac.uk
#
import numpy as np
from statsmodels.tsa.stattools import acf
#
def autoCorrFunc(u,nlags_=1,fft_=False,plot=False):
    """
    Computes autocorrelation function, ACF

    Args:
       `u`    : numpy array of size n, time series samples
       `nlag_`: int>0, max number of lags to include in ACF

    Returnss:
       acf_ : numpy array of size nlags_-1, ACF
    """
    acf_=acf(u,fft=fft_,nlags=nlags_)#,alpha=0.05)
    return acf_

def crossCovFunc(x,y,maxLag):
    """
    Cross-covariance between `x` and `y` considering up to `maxLag` lags.

    Args:
       `x`: 1d numpy array of size n, samples of x
       `y`: 1d numpy array of size n, samples of y
       `maxLag`: int, maximum lag to be considered in cov(SME[x],SME[y])

    Returns:
       `R1`: 1d numpy array of len(`maxLag`)-1, gamma_xy(k), k>0
       `R2`: 1d numpy array of len(`maxLag`)-1, gamma_xy(-k), k>0
    """
    n=len(x)
    R1=[np.cov(np.vstack((x,y)))[0,1]]  #gamma_xy(k), k>0
    R2=[np.cov(np.vstack((x,y)))[0,1]]  #gamma_xy(-k), k>0

    for k in range(1,maxLag): #max range: before cross covariances become oscillatory
        rho1=np.cov(np.vstack((x[:-k],y[k:])))[0,1]
        rho2=np.cov(np.vstack((x[k:],y[:-k])))[0,1]
        R1.append(rho1)
        R2.append(rho2)

    R1=np.asarray(R1)
    R2=np.asarray(R2)
    return R1,R2
