# Tools for forecasting time series by VAR and esitmating their interactions.
#
# https://github.com/simFlowDataLab/PairwiseValidation_turbulenceTS
# Saleh Rezaeiravesh, saleh.rezaeiravesh@manchester.ac.uk
#
import sys 
import numpy as np
sys.path.append('./modules/')
from acf_ccf import autoCorrFunc, crossCovFunc
from var import var_synthesizer,var_train
from itAnalyses import dualInteract,TE_dual_KSG,MI_dual_KSG,embDelayOptimisation
#
def dualInteracts_SingleTSPair(fTS,ncf,lagList,delayList,opts):
    """
    Compute linear and information-theoretic interactions between a single pair of time series `fTS`

    Args: 
       `fTS`: numpy array n x 2, samples of a pair of time series
       `ncf`: int, maximum lag when computing ACF and CCF
       `lagList`: list of int, lags considered when searching for maximum  TE, and MI
       `delayList`: list of int, candidate embedding delays (>=1) over which optimal delay is found
       `opts`: dict, dictionary of options incl. the foreacst info, the TE/MI estimator

    Returns:
       `corrCoef`: scalar, correlation coefficient between time seriers
       `acf`: numpy array of size ncf x 2: ACF for the two time series
       `ccf`: numpy array of size ncf x 2: CCF from TS1->TS2 and TS2->TS1
       `out_te`: dict, TE between the time series par at lagLists
       `out_mi`: dict, MI between the time series par at lagLists
       `esimt_`: string, 'KSG', the estimator method for TE & MI
    """
    n = fTS.shape[0]
    nLag = len(lagList)

    acf = np.zeros((ncf,2))
    ccf = np.zeros((ncf,2))

    corrCoef = np.corrcoef(fTS[:,0],fTS[:,1])[0,1]

    # (1) Compute ACF and CCF for the forecast time seris
    for j in range(2):   #bi-variate TS
        acf[:,j] = autoCorrFunc(fTS[:,j],ncf-1,fft_=False,plot=False)

    ccf[:,0], ccf[:,1] = crossCovFunc(fTS[:,0],fTS[:,1],ncf)

    #(2) Optimise embedding delay for estimating TE
    if 'delayOptim' in opts.keys() and opts['delayOptim']:
       if len(delayList)>0:
          #print('       Optimising embedding delay to be used for TE estimates.') 
          #compute AMIF for x and y over delayList and select optimal delay_x, delay_y
          mi_xx,mi_yy,embDelayX_,embDelayY_ = embDelayOptimisation(fTS,delayList,k_ksg=opts['k_KSG'],threshold=np.exp(1))
          amif = [mi_xx,mi_yy] #Auto-mutual information function
       else:    
          raise ValueError(f'Length of delayList must be >0 for optimising delay.') 
    else:  #set the default embedding delays for source and targe =1
       embDelayX_ = 1
       embDelayY_ = 1
       amif=[[],[]]
    embDelay_ = [embDelayX_,embDelayY_]

    #(3) Compute information theoretic interactions between forecast time series
    ## TE & MI
    if opts['te_mi_estimator'] == 'KSG':
       ### KSG method
       k_KSG = opts['k_KSG']
       out_te = TE_dual_KSG(fTS,lagList,k=k_KSG,embDelay0=embDelayX_,embDelay1=embDelayY_)
       out_mi = MI_dual_KSG(fTS,lagList,k=k_KSG)
       estim_ = 'ksg' 
    else:
       raise ValueError('Invalid te_mi_estimator.') 

    return corrCoef,acf,ccf,out_te,out_mi,estim_,embDelay_,amif

def dualInteracts_forecastData(nTrain,fPred,tsID,nPred,nRep,ncf,lagList,delayList,opts):
    """
    Compute various interactions between pairs of time series in the system in the forecast data by VAR model. 

    Args:
       `nTrain`: int, number of training samples used to create forecast `fPred`
       `fPred`: numpy array nRep x nPred x 2, forecast of a pair of ime series 
       `tsID`: y-index corresponding to `indices`, the wall-normal index assoc. the two interacting time series 
       `nPred`: int, forecasting size (number of samples per time series)
       `nRep`: int, number of repetition of forecast
       `ncf`: int, maximum lag when computing ACF and CCF
       `lagList`: list of int, lags considered when searching for maximum  TE, and MI
       `delayList`: list of int (>0), candidate embedding delays over which optimal delay is found, 
                    if opts['delayOptim']==True.
                    For default embedding delays, pass delayList[1]
       `opts`: dict, dictionary of options incl. the foreacst info, the TE/MI estimator

    Returns:
       `out`: dict, summary of the output of foreacst models and interactions between two time series
       corresponding to tsID. 
    """
    nLag = len(lagList)
    nDelay = len(delayList)

    acf = np.zeros((nRep,ncf,2))
    ccf = np.zeros((nRep,ncf,2))
    mi = np.zeros((nRep,nLag,2))
    te = np.zeros((nRep,nLag,2))
    corrCoef = np.zeros(nRep)
    amif = np.zeros((nRep,nDelay,2))
    embDelay = np.zeros((nRep,2))

    for i in range(nRep):  #repeasted forecast
        if i%5==0:
           print("    ...... Interactions iY=(%d,%d), %d of %d " %(tsID[0],tsID[1],i,nRep))

        #(2) Compute all linear and nonlinear interactions between the pair of time series
        corrCoef_,acf_,ccf_,out_te_,out_mi_,estim_,embDelay_,amif_ = dualInteracts_SingleTSPair(fPred[i,:,:],ncf,lagList,delayList,opts) 

        #(3) Store information-theoretic measures for repeated forecasts in arrays
        corrCoef[i] = corrCoef_
        acf[i,:,:] = acf_
        ccf[i,:,:] = ccf_

        mi[i,:,0] = out_mi_['MI_xy_arr_'+estim_]
        mi[i,:,1] = out_mi_['MI_yx_arr_'+estim_]

        te[i,:,0] = out_te_['TE_xy_arr_'+estim_]
        te[i,:,1] = out_te_['TE_yx_arr_'+estim_]

        if nDelay > 1: #optimal embedding delays have been computed            
           amif[i,:,0] = amif_[0]
           amif[i,:,1] = amif_[1]
           embDelay[i,0] = embDelay_[0]
           embDelay[i,1] = embDelay_[1]
        else:  #if default embedding Delay of 1 i used in TE estimates
           embDelay[i,0] = 1
           embDelay[i,1] = 1

    #(4) Combine the outputs as a dict
    #(a)information theoretic options
    itOpts = {'te_mi_estimator':opts['te_mi_estimator']}
    if opts['te_mi_estimator'] == 'KSG':
       itOpts.update({'k_KSG':opts['k_KSG']}) 

    #(b)forecast options
    frcstOpts = {'forecast_method':opts['forecast_method'],
                 'nTrain':nTrain,
                 'forecastSize':nPred,
                 'forecastRepeat':nRep}    

    if opts['forecast_method'] == 'VAR':
        frcstOpts.update({'VAR_order':opts['pAR'],
                          'VAR_coefs':opts['coefs'].T,
                          'VAR_C_eps':opts['C_eps']
                         })

    #(c)overall database
    out={'info':{'type':'forecast'},
         'tsID':tsID,              #y-location id (in channel flow) of the two time series
         'frcst_opts':frcstOpts,   #forecast options & info (e.g. VAR coefficients, ...)
         'it_opts':itOpts,         #information-theoretic options used
         'lagList':lagList,        #list of lags at which TE, MI are estimated
         'delayList':delayList,    #list of candidate embedding delays (default=[1])
         'AMIF':amif,              #Non-zero arrays of MI_xx,MI_yy only if it_opts['delayOpt']==True
         'embDelay':embDelay,      #default [1,1], otherwise it contains optimal embedding delays
         'ncf':ncf,                #max time step when estimating ACF and CCF 
         'MI':mi,                  #array of Mutual information 
         'TE':te,                  #array of Transfer Entropy (x->y and y->x combined)
         'ACF':acf,                #Autocorrelation function
         'CCF':ccf,                #Cross-covariance function
         'corrCoef':corrCoef       #correlation coefficient
        }

    return out
#
def KVariateForecasts(fTrain,nPred,nRep,opts):
    """
    1. Train a VAR of order p for K-variate time series using `fTrain`
    2. Forecast `nPred` samples from the trained models for K time series. 
       This is repeated `nRep` times due to the stochasticity in the models. 

    Args:
       `fTrain`: numpy array nTrain x K, training samples of time series
       `nPred`: int, forecasting size (number of samples per time series)
       `nRep`: int, number of repetition of forecast
       `opts`: dict, dictionary of options incl. the foreacst info, the TE/MI estimator. 
               NOTE: The input dict is updated with the forecast model's information

    Returns:
       `opts`: dict, input opts updated with asummary of the output of foreacst models and forecast time series samples
       `fPred`: numpy array (nRep, nTrain+nPred,K), samples of respeated forecast for K-VAR(p)
    """
    nTrain = fTrain.shape[0]
    K = fTrain.shape[1]  #number of variates in the collection of time series 

    #K-variate traing & forecast
    fPred = np.zeros((nRep,nPred+nTrain,K))

    if opts['forecast_method'] == 'VAR':

       #Train K-variate VAR(p)
       varCoefs, C_eps_var, A_var = var_train(fTrain,opts['pAR'])
       opts.update({'coefs':varCoefs,'C_eps': C_eps_var})

       print('    Forecast info:')
       print('\t Forecast model:',opts['forecast_method'])
       print('\t Training data (size x K):',fTrain.shape)
       print('\t Size of VAR Coefficient matrix:',varCoefs.shape)
       print('\t Size of VAR Noise Cov Matrix:',C_eps_var.shape)

       #Forecast nRep times of the K-variate model
       for i in range(nRep):  #Repeated forecast
           if i%5==0:
               print("    ... Forecast repeat %d of %d " %(i,nRep))
           fPred_ = var_synthesizer(nPred,fTrain,opts['coefs'].T,opts['C_eps'])
           #Note: fPred_ still includes the fTrain samples (discarded when computing interactions)
           fPred[i,:,:] = fPred_
    else:
       raise ValueError("No method else than VAR is implemented!") 

    return fPred,opts
#
def KVariateForecasts_dualInteracts(fTrain,iLocList,nPred,nRep,ncf,lagList,delayList,opts):
    """
    1. Train a VAR of order p for K-variate time series using `fTrain`
    2. Forecast `nPred` samples from the trained models for K time series. 
       This is repeated `nRep` times due to the stochasticity in the models. 
    3. Call dualInteracts_forecastData() to estimate dual interactions between each pair of 
       forecast time series.

    Args:
       `fTrain`: numpy array nTrain x K, training samples of time series
       `iLocList`: List of y-location indices the time series at which are foreacst and analyzed. 
       `nPred`: int, forecasting size (number of samples per time series)
       `nRep`: int, number of repetition of forecast
       `ncf`: int, maximum lag when computing ACF and CCF
       `lagList`: list of int, lags considered when searching for maximum TE and MI
       `delayList`: list of int (>0), candidate embedding delays over which optimal delay is found, 
                    if opts['delayOptim']==True.
                    For default embedding delays, pass delayList[1]
       `opts`: dict, dictionary of options incl. the foreacst info, the TE/MI estimator. 
               NOTE: The input dict is updated with the forecast model's information

    Returns:
       `out`: dict, summary of the output of foreacst models and interactions between each pair 
              of time series in the K-variate system
    """
    nTrain = fTrain.shape[0]
    K = fTrain.shape[1]  #number of variates in the collection of time series (=len(iLocList))

    #K-variate traing & forecast
    fPred = np.zeros((nRep,nPred+nTrain,K))

    if opts['forecast_method'] == 'VAR':

       #Train K-variate VAR(p)
       varCoefs, C_eps_var, A_var = var_train(fTrain,opts['pAR'])
       opts.update({'coefs':varCoefs,'C_eps': C_eps_var})

       print('    Forecast info:')
       print('\t Forecast model:',opts['forecast_method'])
       print('\t Training data (size x K):',fTrain.shape)
       print('\t Size of VAR Coefficient matrix:',varCoefs.shape)
       print('\t Size of VAR Noise Cov Matrix:',C_eps_var.shape)

       #Forecast nRep times of the K-variate model
       for i in range(nRep):  #Repeated forecast
           fPred_ = var_synthesizer(nPred,fTrain,opts['coefs'].T,opts['C_eps'])
           #Note: fPred_ still includes the fTrain samples (discarded when computing interactions)
           fPred[i,:,:] = fPred_
           if i%5==0:
               print("    ... Forecast repeat %d of %d " %(i,nRep))
    else:
       raise ValueError("No method else than VAR is implemented!") 

    #compute dual interactions between pairs of tim series forecast by the K-variate model
    out_frcst_all = {}
    for i1 in range(len(iLocList)):
        for i2 in range(len(iLocList)):
            if i1 < i2:
               locID = [iLocList[i1],iLocList[i2]]
               indices = [i1,i2]
               fPred1_ = fPred[:,nTrain:,indices[0]]   
               fPred2_ = fPred[:,nTrain:,indices[1]]   
               fPred_ = np.stack((fPred1_,fPred2_),axis=-1)
               out_frcst_ = dualInteracts_forecastData(fTrain,fPred_,locID,nPred,nRep,ncf,lagList,delayList,opts)
               out_frcst_all.update({str(locID[0])+'_'+str(locID[1]):out_frcst_})
    
    return out_frcst_all
#
def simulationData_dualInteracts(fSim,tsID,ncf,lagList,delayList,opts):
    """
    Estimate dual interactions between one pair of simulation time series

    Args:
       `fSim`: numpy array nSim x 2, samples of a pair of time series from CFD simulations
       `tsID`: y-index corresponding to `indices`, the wall-normal index assoc. the two interacting time series 
       `ncf`: int, maximum lag when computing ACF and CCF
       `lagList`: list of int, lags considered when searching for maximum TE, and MI
       `delayList`: list of int (>0), candidate embedding delays over which optimal delay is found, 
                    if opts['delayOptim']==True.
                    For default embedding delays, pass delayList[1]
       `opts`: dict, dictionary of options incl. the foreacst info, the TE/MI estimator

    Returns:
       `out`: dict, summary of interactions between two time series corresponding to tsID. 
    """
    nSim = fSim.shape[0]
    nLag = len(lagList)
    nDelay = len(delayList)

    mi = np.zeros((nLag,2))
    te = np.zeros((nLag,2))
    amif = np.zeros((nDelay,2))

    #(1) Compute all linear and nonlinear interactions between the pair of time series
    corrCoef, acf, ccf, out_te, out_mi, estim_,embDelay_,amif_ = dualInteracts_SingleTSPair(fSim,ncf,lagList,delayList,opts)
 
    #(2) Store information-theoretic measures for repeated forecasts in arrays
    mi[:,0] = out_mi['MI_xy_arr_'+estim_]
    mi[:,1] = out_mi['MI_yx_arr_'+estim_]

    te[:,0] = out_te['TE_xy_arr_'+estim_]
    te[:,1] = out_te['TE_yx_arr_'+estim_]

    if nDelay > 1:  #optimisation for embedding delay has been done
       amif[:,0] =amif_[0]
       amif[:,1] =amif_[1]

    #(3) Combine the outputs as a dict
    #(a)information theoretic options
    itOpts = {'te_mi_estimator':opts['te_mi_estimator']}
    if opts['te_mi_estimator'] == 'KSG':
       itOpts.update({'k_KSG':opts['k_KSG']}) 
    
    #(b)overall database
    out={'info':{'type':'simulation'},
         'tsID':tsID, 
         'it_opts':itOpts,
         'lagList':lagList,
         'delayList':delayList,  
         'AMIF':amif,  #non-zero array of MIxx,MIyy only if it_opts['delayOpt']==True
         'embDelay':embDelay_, #default [1,1], otherwise it contains optimal embedding delays
         'ncf':ncf,
         'MI':mi,
         'TE':te,
         'ACF':acf,
         'CCF':ccf,
         'corrCoef':corrCoef
        }

    return out

