# Main code for computing pairwise interactions within simulation and synthetic time series using MPI
# Note: There is a load balancing (size and number of pairs do not need to match)
# To Run: mpirun -np 3 python -u main_interact_mpi.py
#---------------------------------------------------------------------------------------------
# https://github.com/simFlowDataLab/PairwiseValidation_turbulenceTS
# Saleh Rezaeiravesh, saleh.rezaeiravesh@manchester.ac.uk
#---------------------------------------------------------------------------------------------
#
import sys
import os
from mpi4py import MPI
import numpy as np
from itertools import combinations
import pickle
import time
# local modules
sys.path.append('./modules/')
from dataRead import channelTS_read, channelDataSelect
from trainForecastITtools import simulationData_dualInteracts, dualInteracts_forecastData, KVariateForecasts
#
#
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Disable output buffering
sys.stdout.flush()
os.environ["PYTHONUNBUFFERED"] = "1"
##
print(f">>> Rank {rank} of {size} started.", flush=True)
#
# ---------------------------------------------------------------------------------
# -----SETTINGS -------------------------------------------------------------------
#
## Path to the channel flow time series data
dataPath = './data/chan300/'

## Data extraction: selection of time series for measuring their interactions
#iLocList = [10,40]          # list of indices of wall-normal points, 0: wall, 63: channel centre
iLocList = np.arange(1,64,3) # list of indices of wall-normal points, 0: wall, 63: channel centre
iDiscard = 0  # number of samples to be discarded from the beginning of data
iSkip = 10    # Down-sampling frequency

## Forecast settings
nRep = 50      # number of repetition of generating samples by VAR model
nPred = 10437  # number of time samples to synthesize by VAR model
synthetic_method = 'VAR'  
pAR = 60       # order of the VAR model

## Inputs for computing interaction measures
lagList = [1,2,3,4,5,8,10,15]   # list of lags at which TE and MI are computed (also to find maxTE, maxMI)
ncf = 1000                      # max lag in autocorrelation and cross-covariance function
te_mi_estimator = 'KSG'         # 'KSG', method for estimating TE and MI
k_KSG = 10         # value of k in KSG esitmator of TE and MI
delayOptim = True  # if True, optimal delay for source & target for TE estimates are computed

if delayOptim: 
   #list of candidate delays to search for optimal embedding  delay  
   #optimisation method: using AMIF
   delayList = [int(i) for i in range(1,15)] #List of candidate embedding delays
   delayList += [int(i) for i in range(15,105,5)] 
else:
   delayList = [1]   #default embedding delay for source and target

## Output data
caseName = 'mpi_test'               # output case name
outDir = './outData/'+caseName+'/'  # output directory where the output pickle files are written
#
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
#
# (0) Add the channel flow centre to iLocList
if 63 not in iLocList:
    iLocList = np.append(iLocList, 63)
print('iLocList:', iLocList, flush=True)

# (1) Create a list of dual interactions between iLocList elements
#
indexList_pairs = list(combinations(np.arange(len(iLocList)), 2))
nPairs = len(indexList_pairs)
print('indexList_pairs:', indexList_pairs, flush=True)
print('Number of pairs:', nPairs, flush=True)

# (2) Read channel flow time series data
#
t, y, ypls, uTau, u = channelTS_read(dataPath)
print('y+ at iLocList:', ypls[iLocList], flush=True)

# (3) Construct multivariate TS data at iLocList
#
fSim = channelDataSelect(iLocList=iLocList, u=u, iDiscard=iDiscard, iSkip=iSkip)
print('Array of simulation data:', fSim.shape, flush=True)

# (4) Select training data
#
fTrain = fSim
nTrain = fTrain.shape[0]
print('Array of training data:', fTrain.shape, flush=True)

# (5) dict of options for VAR-synthesis and interaction estimates
#
opts_it = {'te_mi_estimator': te_mi_estimator, 'k_KSG': k_KSG,
           'forecast_method': synthetic_method, 'pAR': pAR,
           'delayOptim':delayOptim}

# (6) Safety check: proceed even if size < nPairs but warn (we will still distribute)
#
if size < 1:
    raise ValueError(f'Invalid number of MPI ranks: {size}')

# (7) DISTRIBUTE PAIRS to RANKS (balanced)
#
all_pair_indices = np.arange(nPairs)
pair_indices_split = np.array_split(all_pair_indices, size)
pairs_local_indices = pair_indices_split[rank]  # numpy array, possibly empty
pairs_local_indices = [] if pairs_local_indices.size == 0 else pairs_local_indices.tolist()
print(f"[Rank {rank}] Assigned pair idxs: {pairs_local_indices}", flush=True)

# (8) Interactions between pairs of simulation time series
#
out_sim_local = []  # collect results for the pairs this rank processes

if len(pairs_local_indices) > 0:
   if rank==0: print('#---------- Dual Interactions - Simulation TS -------------')
   print(f'... Simulation interactions started - rank {rank} (processing {len(pairs_local_indices)} pair)!', flush=True)
else:
   print(f'... No simulation pairs assigned to rank {rank}.', flush=True)

t_start_all = time.time()
for ip in pairs_local_indices:
    pair_index = indexList_pairs[int(ip)]
    I1, I2 = pair_index[0], pair_index[1]
    locID = [int(iLocList[I1]), int(iLocList[I2])]
    print(f"    [Rank {rank}] Processing simulation pair idx {ip} -> locID {locID}", flush=True)

    t0 = time.time()
    # form pair data for this rank locally (small 2-column array)
    fSim_pair = np.column_stack((fSim[:, I1], fSim[:, I2]))  # shape (nSamples, 2)
    out_sim_ = simulationData_dualInteracts(fSim_pair, locID, ncf, lagList, delayList,opts_it)
    t1 = time.time()
    print(f"    [Rank {rank}] Completed simulation pair {locID}. Elapsed time={t1 - t0:.2f}s", flush=True)

    # append
    out_sim_local.append(out_sim_)

t_end_all = time.time()
print(f"    [Rank {rank}] Simulation interactions done (local). Total elapsed {t_end_all - t_start_all:.2f}s", flush=True)

# (b) Gather all simulation results to rank 0 and merge there
all_outSim_lists = comm.gather(out_sim_local, root=0)

if rank == 0:
    # Flatten lists and merge by key
    all_outSim_flat = [item for sublist in all_outSim_lists for item in sublist if item is not None]
    out_sim_all = {}
    for item in all_outSim_flat:
        locID_ = item['tsID']  # assuming your function sets 'tsID'
        key = f"{locID_[0]}_{locID_[1]}"
        out_sim_all[key] = item

    # add info
    yplsLogList = ypls[iLocList]
    out_sim_all.update({'iLocList': iLocList, 'yplsLocList': yplsLogList, 'lagList': lagList, 'ncf': ncf,'delayList':delayList})

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    with open(os.path.join(outDir, 'outSimInteract_multiLocs'), 'wb') as F:
        pickle.dump(out_sim_all, F)
    print(f'... Writing simulation-interaction results at {outDir} completed - rank {rank}!', flush=True)

# Ensure all ranks wait until simulation results are gathered/written
comm.Barrier()

# (9) Forecast samples from K-variate VAR(p) on rank==0 (to ensure the same samples on all ranks)
#
if rank == 0:
    print('#---------- Forecast by K-variate VAR(p) -------------')
    print(f'... VAR synthesis started (single rank) - rank {rank}!', flush=True)
    t0 = time.time()
    fPred, opts_it = KVariateForecasts(fTrain, nPred, nRep, opts_it)
    t1 = time.time()
    print(f'    Completed on rank {rank}. Elapsed time={t1 - t0:.2f}s', flush=True)
else:
    fPred = None
    # Note: opts_it will be broadcast as well below

# (10) Broadcast the VAR synthetic samples to all ranks
#
fPred = comm.bcast(fPred, root=0)
opts_it = comm.bcast(opts_it, root=0)
comm.Barrier()
print(f"    [Rank {rank}] Received broadcasted synthetic data (fPred shape: {None if fPred is None else fPred.shape}).", flush=True)

# (11) Interactions between synthetic time series
#
out_pred_local = []

if len(pairs_local_indices) > 0:
    if rank==0: print('#---------- Dual Interactions - Forecast TS -------------')
    print(f'... Forecast interactions started - rank {rank}!', flush=True)
else:
    print(f'... No synthetic pairs assigned to rank {rank}.', flush=True)

t_start_pf = time.time()
for ip in pairs_local_indices:
    pair_index = indexList_pairs[int(ip)]
    I1, I2 = pair_index[0], pair_index[1]
    locID = [int(iLocList[I1]), int(iLocList[I2])]

    print(f"    [Rank {rank}] Processing synthetic pair idx {ip} -> locID {locID}", flush=True)

    t0 = time.time()
    # build synthetic pair array: fPred has shape (nRep, nTotal, nVar)
    # original code used fPred[:, nTrain:, I1] etc. Keep that logic
    fPred_pair = np.stack((fPred[:, nTrain:, I1], fPred[:, nTrain:, I2]), axis=-1)  # shape (nRep, nPred, 2)
    out_pred_ = dualInteracts_forecastData(nTrain, fPred_pair, locID, nPred, nRep, ncf, lagList,delayList, opts_it)
    t1 = time.time()
    print(f"    [Rank {rank}] Completed synthetic pair {locID}. Elapsed time={t1 - t0:.2f}s", flush=True)

    out_pred_local.append(out_pred_)

t_end_pf = time.time()
print(f"    [Rank {rank}] Forecast interactions done (local). Total elapsed {t_end_pf - t_start_pf:.2f}s", flush=True)

# Gather synthetic results to rank 0 and merge
all_outPred_lists = comm.gather(out_pred_local, root=0)

if rank == 0:
    all_outPred_flat = [item for sublist in all_outPred_lists for item in sublist if item is not None]
    out_frcst_all = {}
    for item in all_outPred_flat:
        locID_ = item['tsID']
        key = f"{locID_[0]}_{locID_[1]}"
        out_frcst_all[key] = item

    # add info
    yplsLogList = ypls[iLocList]
    out_frcst_all.update({'iLocList': iLocList, 'yplsLocList': yplsLogList, 'lagList': lagList, 'ncf':ncf,'delayList':delayList})

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    with open(os.path.join(outDir, f'outFrcstInteract_multiLocs_{synthetic_method}'), 'wb') as F:
        pickle.dump(out_frcst_all, F)
    print(f'... Writing synthetic TS-interaction results at {outDir} completed - rank {rank}!', flush=True)

# (12) finalize MPI
#
comm.Barrier()

if rank == 0:
    print("All ranks finished successfully.", flush=True)

MPI.Finalize()
#
