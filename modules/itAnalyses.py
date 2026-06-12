# Functions used to compute interactions between time series
# Interactions: TE, MI, CCF, correlation coefficient
#
# https://github.com/simFlowDataLab/PairwiseValidation_turbulenceTS
# Saleh Rezaeiravesh, saleh.rezaeiravesh@manchester.ac.uk
#
import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
sys.path.append('./')
from mi import mi_multiLag
from te import bi_te_multiLag
#
def TE_dual_KSG(f,lagList,k=3,embDelay0=1,embDelay1=1):
    """
    Compute TE between two time series at lags in `lagList` - KSG Method

    Args:
       `f`: array of shape (number of samples x 2), samples of two time series
       `lagList`: list of integers, set of lags to be considered in the interactions
       `k`:int, k in KNN used in the KSG estimators (only applicable to KSG)
       `embDelay0`: int (>=1), embedding delay of time series 1 (f[:,0]) (default=1)
       `embDelay1`: int (>=1), embedding delay of time series 2 (f[:,1]) (default=1)

    Returns:
       `out`: dict, containing `maxTE_xy_ksg`, `maxTE_yx_ksg`, `TE_xy_arr_ksg`, `TE_yx_arr_ksg`
    """
    out={}
    maxTE_xy, TE_xy_arr = bi_te_multiLag(f[:,0],f[:,1],lagList,embDelayX=embDelay0,embDelayY=embDelay1).ksg(k=k)
    maxTE_yx, TE_yx_arr = bi_te_multiLag(f[:,1],f[:,0],lagList,embDelayX=embDelay1,embDelayY=embDelay0).ksg(k=k)
    out.update({'maxTE_xy_ksg':maxTE_xy, 'maxTE_yx_ksg':maxTE_yx, 
                'TE_xy_arr_ksg':TE_xy_arr, 'TE_yx_arr_ksg':TE_yx_arr})
    return out

def MI_dual_KSG(f,lagList,k=3):
    """
    Compute MI between two time series at lags in `lagList` - KSG method

    Args:
       `f`: array of shape (number of samples x 2), samples of two time series
       `lagList`: list of integers, set of lags to be considered in the interactions
       `k`:int, k in KNN used in the KSG estimators (only applicable to KSG)

    Returns:
       `out`: dict, containing `maxMI_xy_kde`, `maxMI_yx_kde`, `MI_xy_arr_kde`, `MI_yx_arr_kde`
    """
    out={}
    MI_xy_arr, maxMI_xy, _ = mi_multiLag(f[:,0],f[:,1],lagList).ksg(k=k)
    MI_yx_arr, maxMI_yx, _ = mi_multiLag(f[:,1],f[:,0],lagList).ksg(k=k)
    out.update({'maxMI_xy_ksg':maxMI_xy, 'maxMI_yx_ksg':maxMI_yx, 
                'MI_xy_arr_ksg':MI_xy_arr, 'MI_yx_arr_ksg':MI_yx_arr})
    return out

def dualInteract(f,lagList,k_ksg=3):
    """
    Compute maximum TE, MI, and correlation coefficient between two time series

    Args:
       `f`: array of shape (number of samples x 2), samples of two time series
       `lagList`: list of integers, set of lags to be considered in the interactions
       `k_ksg`:int, k in KNN used in the KSG estimators (only applicable to KSG)

    Return:
       `out`: dict, containing maxpTE and pTE_arr, maxTE_xy, TE_xy_arr (by KDe and KSG methods), corrCoef 
    """
    out = {}

    #maximum TE & TE array at lags in the lagList - KSG method
    out_teKSG = TE_dual_KSG(f,lagList,k=k_ksg)
    out.update(out_teKSG)

    #maximum MI - KSG method
    out_miKSG = MI_dual_KSG(f,lagList,k=k_ksg)
    out.update(out_miKSG)

    #correlation coefficient
    corrCoef = np.corrcoef(f[:,0],f[:,1])[0,1]
    out.update({'corrCoef':corrCoef})

    return out
#
def delay_selector(amif,delayList,threshold=np.exp(1)):
    """
    Selecting optimal embedding delay (tau) from AMIF function `amif` computed over list `delayList`

    Args:
       `amif`: array of size len(delayList): Auto-mutual information function at `delayList`
       `delayList`: list, candidate delays
       `threshold`:float, threhold for selecting optimal delay when AMIF is decaying with no minimum (default=e)

    returns 
      `id_min`: int, index associated to optimal tau (tau_optimal = delayList[id_min])
    """
    id1 = np.argmin(amif)  #where min(amif) is observed
    id2 = np.where(amif<amif[0]/threshold)[0]  #where amif drops below threshold
    #select the first delay at which amif drops below threshold. This is for the case 
    # where the AMIF is decaying without any minimum and reaches a plateau at large delays
    if len(id2)>1:
       id2 = np.min(id2)  
    id_min = min(id1,id2)
    return id_min

def embDelayOptimisation(f,delayList,k_ksg=3,threshold=np.exp(1)):
    """
    Finding optimal embedding delay for esimtating TE between f[:,0] and f[:,1] using candidate delays in `delayList`.
    The optimisation is based on AMIF (Auto-Mutual Information Function) of f[:,0] and f[:,1]

    Args:
       `f`: array of shape (number of samples x 2), samples of two time series
       `delayList`: list of int, candidate embedding delays (>=1) over which optimal delay is found
       `k_ksg`:int, k in KNN used in the KSG estimators (only applicable to KSG)
       `threshold`:float, threhold for selecting optimal delay when AMIF is decaying with no minimum (default=e)

    Return:
       `mi_xx`: numpy array of length `len(delayList)`, AMIF(f[:,0]) over `delayList`
       `mi_yy`: numpy array of length `len(delayList)`, AMIF(f[:,1]) over `delayList`
       `embdelayX`: int, optimal embedding delay for f[:,0]
       `embdelayY`: int, optimal embedding delay for f[:,1]
    """
    #compute AMIF for x and y over delayList
    mi_xx, _, _ = mi_multiLag(f[:,0],f[:,0],delayList).ksg(k=k_ksg)
    mi_yy, _, _ = mi_multiLag(f[:,1],f[:,1],delayList).ksg(k=k_ksg)

    #select optimal delays based in AMIFxx and AMIFyy
    idx = delay_selector(mi_xx,delayList,threshold=threshold)
    idy = delay_selector(mi_yy,delayList,threshold=threshold)
    embDelayX = delayList[idx]
    embDelayY = delayList[idy]

    return mi_xx,mi_yy,embDelayX,embDelayY
