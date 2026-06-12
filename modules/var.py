#
# https://github.com/simFlowDataLab/PairwiseValidation_turbulenceTS
# Saleh Rezaeiravesh, saleh.rezaeiravesh@manchester.ac.uk
#
import numpy as np
from statsmodels.tsa.api import VAR
#
def var_synthesizer(n,xInit,varCoefs,varCov):
    """
    Synthesize samples from a VAR model with known coefficients and noise covariance matrix

    VAR order: p
    Number of variates: K

    Args:
       `xInit`: numpy array of shape (nInit x K)
       `varCoefs`: numpy array of shape (K x p*K+1), VAR coefficients
       `carCov`: Numpy array of shape (K x K) , VAR noise covariance matrix

    Returns 
       `x`: numpy array of shape ((n+nInit) x K), samples from the VAR model
    """
    nVar=varCoefs.shape[0]   #number of variables
    nCoefs=varCoefs.shape[1]   #number of coefficients per variable
    nInit=xInit.shape[0]    #number of initial samples, should be greater than p
    p=int((nCoefs-1)/nVar)   #order of the VAR

    x=np.zeros((nInit+n,nVar))
    x[:nInit,:]=xInit

    #rearrange VAR coefficients (lag>1) in matrices A1, A2, ..., Ap where Ai is nVar x nVar
    A=[]
    for k in range(p):
        K1=k*nVar+1
        K2=K1+nVar
        A.append(varCoefs[:,K1:K2])
    A=np.asarray(A)

    #Predict samples
    for i in range(nInit,n+nInit):
        noiseSample=np.random.multivariate_normal(np.zeros(nVar),varCov)
        for k in range(p):
            x[i,:]+=A[k,:,:]@x[i-k-1,:]
        x[i,:]+=varCoefs[:,0]+noiseSample

    return x

def var_train(fTrain,p,verbose=False):
    """
    Training a VAR for two time series and using it for forecasting

    Args: 

       `fTrain`: numpy array (nTrain, 2), training time samples
       `p`: int, order of the VAR model
    """

     # Train VAR using the training data and a given model order
    var_=VAR(fTrain)
    results=var_.fit(p)

    c=results.summary()
    if verbose:
       print(c)

    # Estimated coefficients of the VAR: columns of the following tensor are the coefficients
    #  of the ARMs in the VAR model.
    varCoefs=results.params

    # Noise covariance matrix between the variables in the VAR model
    varCov=results.sigma_u

    #(optional-for poltting only) Rearrange VAR coefficients (lag>1) in matrices A1, A2, ..., Ap where Ai is nVar x nVar
    nVar=varCoefs.shape[-1]   #number of variables

    A=[]
    for k in range(p):
        K1=k*nVar+1
        K2=K1+nVar
        A.append(varCoefs.T[:,K1:K2])
    A=np.asarray(A)

    return varCoefs,varCov,A

