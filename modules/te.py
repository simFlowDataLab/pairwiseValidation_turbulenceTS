# Compute bivariate transfer entropy beween time series
#
# https://github.com/simFlowDataLab/PairwiseValidation_turbulenceTS
# Saleh Rezaeiravesh, saleh.rezaeiravesh@manchester.ac.uk
#
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma, digamma
from sklearn.neighbors import NearestNeighbors
#
class bi_te:    
    """
    Estimate Transfer Entropy (TE) for a pair of time series at a single lag
    """

    def __init__(self,x,y,embDim,embDelayX=1,embDelayY=1):
        """
        Args: 
           `x`: 1d numpy array of size n, containing samples of source time series
           `y`: 1d numpy array of size n, containing samples of target time series
           `embDim`: int (>1), embedding dimension (number of delayed samples)
           `embDelayX`: int (>=1), source's embedding delay (default = 1)
           `embDelayY`: int (>=1), target's embedding delay (default = 1)

        Return:
            `te`: float, transfer entropy (x->y) for embedding samples
        """
        self.x = x
        self.y = y
        self.embDim = embDim
        self.embDelayX = embDelayX
        self.embDelayY = embDelayY

        self.n = self.x.shape[0]
        assert self.y.shape[0] == self.n

    def ksg_embedding(self):
        """
        Construct embeddings for the KSG estimator 
        """
        #Farthest lag to consider
        startX = self.embDim * self.embDelayX
        startY = self.embDim * self.embDelayY
        start = max(startX,startY)

        #Indices for the past of X and Y
        Ix = np.arange(1, self.embDim + 1) * self.embDelayX
        Iy = np.arange(1, self.embDim + 1) * self.embDelayY

        #Embedding space for xPast, yPast, yFuture
        xPast = np.array([self.x[i - Ix] for i in range(start, self.n)])
        yPast = np.array([self.y[i - Iy] for i in range(start, self.n)])
        yFutu = self.y[start:self.n]
        return xPast, yPast, yFutu

    def surrogate_shuffle_xpast(self,xPast, rng):
        """
        Shuffle rows of xPast - to create a surrogate
        """
        perm = rng.permutation(xPast.shape[0])
        return xPast[perm, :]

    def ksg(self,k=3,tol=1e-8,xPastRowShuffle=False,rng=None):    
        """
        Estimate transfer entropy using the KSG-based method (KNN type). 

        TE is defined as a conditional mutual information:        
        TE(x->y) = MI(y,x_past|y_past)

        Reference:
        Frenzel S. & Pompe B., 2007. doi: 0031-9007=07=99(20)=204101(4)

        Args:
           `k`: int, k-th nearest points to each sample
           `tol`: float, tolerance (small value)
           `xPastRowShuffle`: bool, if True, random shuffling of xPast rows to create surrogate
                         suggested by Pinto et al. 2024, doi: 10.3389/fnetp.2024.1385421
           `rng`: numpy array, only used if xPastShuffle is True; example: np.random.default_rng(125)             
        """
        self.k = k
        self.tol = tol

        #Create time series embeddings 
        xPast, yPast, yFutu = self.ksg_embedding()
        assert xPast.shape[0] == yPast.shape[0] == yFutu.shape[0]

        nEff = xPast.shape[0]

        #surrogate construction by random shuffling the rows of xPast to destroy dependence 
        # between xPast with yPast and yFutu while not distorting x, y internal structure
        if xPastRowShuffle:
           if rng is None: 
              raise ValueError('rng should be passed as rng=np.random.default_rng(seed).') 
           else:
               xPast = self.surrogate_shuffle_xpast(xPast, rng)

        #Create joint embeddings
        XZ = np.hstack((yFutu[:,None], yPast))   
        YZ = np.hstack((xPast, yPast))   
        XYZ = np.hstack((yFutu[:,None], xPast, yPast))

        # Create kd trees
        tree_XYZ = cKDTree(XYZ)

        # find distance of each point from its k-th neighbours in the joint set
        r_Z, _ = tree_XYZ.query(XYZ, k + 1, p=np.inf)

        eps_Z = r_Z[:, -1]  - self.tol

        #Count neighbors in marginal spaces (X-space and Y-space)
        knn_YZ = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(YZ)
        knn_XZ = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(XZ)
        knn_Z = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(yPast)

        nYZ, nXZ, nZ = [], [], []
        for i in range(nEff):
            epsi = eps_Z[i]
            yZi = YZ[i].reshape(1, -1)
            xZi = XZ[i].reshape(1, -1)
            yi  = yPast[i].reshape(1, -1)

            nYZ.append(knn_YZ.radius_neighbors(yZi, epsi, return_distance=False)[0])
            nXZ.append(knn_XZ.radius_neighbors(xZi, epsi, return_distance=False)[0])
            nZ.append(knn_Z.radius_neighbors(yi,  epsi, return_distance=False)[0])
        nYZ, nXZ, nZ = map(lambda arr: np.array(arr, dtype=object), [nYZ, nXZ, nZ])    

        # count the number of points in the vicinity of each point, excluding itself
        nZ = np.array([float(len(i) - 1) for i in nZ])
        nYZ = np.array([float(len(i) - 1) for i in nYZ])
        nXZ = np.array([float(len(i) - 1) for i in nXZ])

        te = digamma(k) + np.mean(digamma(nZ + 1) - digamma(nXZ + 1) - digamma(nYZ + 1))
        return te

class bi_te_multiLag:
    """
    Estimate maximum bivariate TE between two time series over a set of candidate lags
    """
    def __init__(self,x,y,lagList,embDelayX=1,embDelayY=1):
        """
        Args: 
           `x`: 1d numpy array of size n, containing samples of source time series
           `y`: 1d numpy array of size n, containing samples of target time series
           `lagList`: List of int (>0), candidate embedding dimensions (lags) 
           `embDelayX`: int (>=1), source's embedding delay (default = 1)
           `embDelayY`: int (>=1), target's embedding delay (default = 1)

        Return:
            `maxTE`: float, maximum element in 
            `teArray`: 1d numpy array of transfer entropy (x->y) for lags in `lagList`
        """
        self.x = x
        self.y = y
        self.embDim = lagList
        self.embDelayX = embDelayX
        self.embDelayY = embDelayY

        self.n = self.x.shape[0]
        assert self.y.shape[0] == self.n
        self.teArray = []

    def ksg(self,k=3,tol=1e-8,xPastRowShuffle=False,rng=None):    
        """
        Estimate transfer entropy using the KSG-based method (KNN type). 

        TE is defined as a conditional mutual information:        
        TE(x->y) = MI(y,x_past|y_past)

        Reference:
        Frenzel S. & Pompe B., 2007. doi: 0031-9007=07=99(20)=204101(4)

        Args:
           `k`: int, k-th nearest points to each sample
           `tol`: float, tolerance (small value)
           `xPastRowShuffle`: bool, if True, random shuffling of xPast rows to create surrogate
                         suggested by Pnto et al. 2024, doi: 10.3389/fnetp.2024.1385421
           `rng`: numpy array, only used if xPastShuffle is True              
        """
        for embDim_ in self.embDim:
            bi_te_ksg_ = bi_te(self.x,self.y,embDim_,self.embDelayX,self.embDelayY).ksg(k=k,tol=tol,
                                                   xPastRowShuffle=xPastRowShuffle, rng=rng)
            self.teArray.append(bi_te_ksg_)
        self.teArray = np.array(self.teArray)    
        self.maxTE = np.max(self.teArray)
        return self.maxTE, self.teArray
