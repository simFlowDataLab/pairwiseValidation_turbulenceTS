# Post-processing the pickle data resulted from `main_interact_mpi.py`
#
# https://github.com/simFlowDataLab/PairwiseValidation_turbulenceTS
# Saleh Rezaeiravesh, saleh.rezaeiravesh@manchester.ac.uk
#
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
#
#----- settings ---------------------------------
# Folder where the pickle files of main_interact_mpi.py are written
caseName = 'mpi_test'  
#------------------------------------------------
#
#directory to save postprocessed data
inDir = './outData/'+caseName+'/'
outDir = inDir + 'postprcssd/' 
#
def pstprcss(qoi,nLoc,N,out_sim,out_frcst):
    """
    Post-process ACF, CCF, TE, MI for simulation and synthetic time series

    N = nLag (for MI, TE) or ncf (for ACF, CCF) or nDelay for AMIF
    """
    data_sim = np.zeros((nLoc,nLoc,N)) 
    data_frcst_m = np.zeros((nLoc,nLoc,N)) 
    data_frcst_s = np.zeros((nLoc,nLoc,N)) 
    maxData_frcst_m = np.zeros((nLoc,nLoc))
    maxData_frcst_s = np.zeros((nLoc,nLoc))

    for i1 in range(nLoc):
        for i2 in range(nLoc):
            if i1 < i2:
               locID = [iLocList[i1],iLocList[i2]]
               locID_str = str(locID[0])+'_'+str(locID[1])
               #print('Postprocess - TS pairs at iy1 = %d, iy2 = %d' %(locID[0],locID[1]))

               #extract all interactions at i1-i2 location
               db_sim = out_sim[locID_str]
               db_frcst = out_frcst[locID_str]

               #select the 'qoi' interactions: 'TE', 'MI' - function of lag
               #simulation data
               data_sim[i1,i2,:] = db_sim[qoi][:,0]
               data_sim[i2,i1,:] = db_sim[qoi][:,1]

               #mean and sdev of of synthetic measures (due to repetitions)
               data_frcst_m[i1,i2,:] = np.mean(db_frcst[qoi][:,:,0],axis=0)
               data_frcst_s[i1,i2,:] = np.std(db_frcst[qoi][:,:,0],axis=0)
               data_frcst_m[i2,i1,:] = np.mean(db_frcst[qoi][:,:,1],axis=0)
               data_frcst_s[i2,i1,:] = np.std(db_frcst[qoi][:,:,1],axis=0)

               if qoi not in ['ACF','CCF']:                   
                  maxData_frcst_ = np.max(db_frcst[qoi][:,:,0],axis=1)
                  maxData_frcst_m[i1,i2] = np.mean(maxData_frcst_)
                  maxData_frcst_s[i1,i2] = np.std(maxData_frcst_)

                  maxData_frcst_ = np.max(db_frcst[qoi][:,:,1],axis=1)
                     
                  #print(qoi,np.argmax(db_frcst[qoi][:,:,0],axis=1),np.argmax(db_frcst[qoi][:,:,1],axis=1))
                  maxData_frcst_m[i2,i1] = np.mean(maxData_frcst_)
                  maxData_frcst_s[i2,i1] = np.std(maxData_frcst_)

    return data_sim,data_frcst_m,data_frcst_s,maxData_frcst_m,maxData_frcst_s        
#
def pstprcss_corrCoef(nLoc,out_sim,out_frcst):
    """
    Post-process correlation coefficient for simulation and synthetic time series
    """
    data_sim = np.zeros((nLoc,nLoc)) 
    data_frcst_m = np.zeros((nLoc,nLoc)) 
    data_frcst_s = np.zeros((nLoc,nLoc)) 
    for i1 in range(nLoc):
        for i2 in range(nLoc):
            if i1 < i2:
               locID = [iLocList[i1],iLocList[i2]]
               locID_str = str(locID[0])+'_'+str(locID[1])

               #extract all interactions at i1-i2 location
               db_sim = out_sim[locID_str]
               db_frcst = out_frcst[locID_str]

               #cross-crrelation of simulation data
               data_sim[i1,i2] = db_sim['corrCoef']

               #cross-correlation of mean and sdev of of synthetic (due to repetitions)
               data_frcst_m[i1,i2] = np.mean(db_frcst['corrCoef'])
               data_frcst_s[i1,i2] = np.std(db_frcst['corrCoef'])
    return data_sim,data_frcst_m,data_frcst_s        
#
def pstprcss_embDelay(nLoc,out_sim,out_frcst):
    """
    Post-process embedding delays used for computing TE
    """
    data_sim = np.zeros(nLoc) #results of simulation data
    locID = [iLocList[0],iLocList[1]]
    locID_str = str(locID[0])+'_'+str(locID[1])
    nRep = db_=out_frcst[locID_str]['embDelay'].shape[0]
    data_frcst = np.zeros((nLoc,nRep)) #results of repeated synthetic data
    for i1 in range(nLoc):
        for i2 in range(nLoc):
            if i1 < i2:
               locID = [iLocList[i1],iLocList[i2]]
               locID_str = str(locID[0])+'_'+str(locID[1])

               #extract all interactions at i1-i2 location
               db_sim = out_sim[locID_str]
               db_frcst = out_frcst[locID_str]

               #cross-crrelation of simulation data
               if i1==0:
                  data_sim[i2] = db_sim['embDelay'][1]
                  if i2==1:
                     data_sim[0] = db_sim['embDelay'][0]

               #cross-correlation of mean and sdev of of synthetic (due to repetitions)
               if i1 == 0:
                  data_frcst[i2,:] = db_frcst['embDelay'][:,1]
                  if i2==1:
                     data_frcst[0,:] = db_frcst['embDelay'][:,0]

    return data_sim,data_frcst
#    
#
#(1) Read pickle data
## (a) interactions between simulation time series
with open(inDir+'outSimInteract_multiLocs','rb') as F:
    out_sim = pickle.load(F)
print('out_sim.keys():',out_sim.keys())
#take common values for sim and var dicts
iLocList = out_sim['iLocList']
yplsLocList = out_sim['yplsLocList']
lagList = out_sim['lagList']
if 'delayList' in out_sim.keys(): #the newer version of the code where delayList is available
    delayList = out_sim['delayList']
    nDelay = len(delayList)  #if ==1: default embedding delays were used (No AMIF)
                             #if >1: optimal embedding delays (from delayList) were adapted 
else:
    nDelay = 0 

ncf = out_sim['ncf']

## (b) interactions between VAR-synthetic time series
with open(inDir+'outFrcstInteract_multiLocs_'+'VAR','rb') as F:
    out_VAR = pickle.load(F)
print('out_VAR.keys():',out_VAR.keys())

#(2) postprocess interactions between time series corresponding iLocList   
nLoc = len(iLocList)
nLag = len(lagList)    

#simulation & VAR
TE_sim,TE_var_m,TE_var_s,maxTE_var_m,maxTE_var_s = pstprcss('TE',nLoc,nLag,out_sim,out_VAR)
MI_sim,MI_var_m,MI_var_s,maxMI_var_m,maxMI_var_s = pstprcss('MI',nLoc,nLag,out_sim,out_VAR)
ACF_sim,ACF_var_m,ACF_var_s,_ ,_ = pstprcss('ACF',nLoc,ncf,out_sim,out_VAR)
CCF_sim,CCF_var_m,CCF_var_s,_ ,_ = pstprcss('CCF',nLoc,ncf,out_sim,out_VAR)    
if nDelay > 1:
   AMIF_sim,AMIF_var_m,AMIF_var_s,maxAMIF_var_m,maxAMIF_var_s = pstprcss('AMIF',nLoc,nDelay,out_sim,out_VAR)
if 'delayList' in out_VAR.keys():   
   embDelay_sim ,embDelay_frcst = pstprcss_embDelay(nLoc,out_sim,out_VAR)

maxTE_sim = np.max(TE_sim,axis=2)
maxMI_sim = np.max(MI_sim,axis=2)

corrCoef_sim, corrCoef_var_m, corrCoef_var_s = pstprcss_corrCoef(nLoc,out_sim,out_VAR)

#(3) Dump the postprocessed data
##(a) collect common information in analyses
locID_ = [iLocList[0],iLocList[1]]
locID_str_ = str(locID_[0])+'_'+str(locID_[1])

##(b) CFD simulation interactions (to be used as reference)
db_sim_ = out_sim[locID_str_]
it_opts_sim = db_sim_['it_opts']

#print('sim',db_sim_.keys())

postOut_sim = {'type':'simulation',
               'it_opts':it_opts_sim,
               'iLocList':iLocList,
               'yplsLocList':yplsLocList,
               'lagList':lagList,
               'ncf':ncf,
               'TE':TE_sim,   #nLoc x nLoc x len(lagList) 
               'MI':MI_sim,   #nLoc x nLoc x len(lagList) 
               'ACF':ACF_sim, #nLoc x nLoc x ncf 
               'CCF':CCF_sim, #nLoc x nLoc x ncf 
               'maxTE':maxTE_sim,    #nLoc x nLoc
               'maxMI':maxMI_sim,    #nLoc x nLoc
               'corrCoef':corrCoef_sim   #nLoc x nLoc
               }
if 'delayList' in out_sim.keys():        
   postOut_sim.update({'delayList':delayList,
                       'embDelay':embDelay_sim #nLoc #embedding delays for computing TE
                       }) 
   if nDelay>1:
      postOut_sim.update({'AMIF':AMIF_sim}) #nLoc x nLoc x len(delayList)
    
##(c) VAR data interactions (averaged over nRep repetitions)
db_var_ = out_VAR[locID_str_]
#print('var',db_var_.keys())

postOut_var = {'type':'VAR_forecast',
               'it_opts':db_var_['it_opts'],
               'frcst_opts':db_var_['frcst_opts'],
               'iLocList':iLocList,
               'yplsLocList':yplsLocList,
               'lagList':lagList,
               'ncf':ncf,
               'TE_mean':TE_var_m,   #nLoc x nLoc x len(lagList) 
               'TE_sdev':TE_var_s,   #nLoc x nLoc x len(lagList) 
               'MI_mean':MI_var_m,   #nLoc x nLoc x len(lagList) 
               'MI_sdev':MI_var_s,   #nLoc x nLoc x len(lagList) 
               'ACF_mean':ACF_var_m, #nLoc x nLoc x ncf
               'ACF_sdev':ACF_var_s, #nLoc x nLoc x ncf
               'CCF_mean':CCF_var_m, #nLoc x nLoc x ncf
               'CCF_sdev':CCF_var_s, #nLoc x nLoc x ncf
               'maxTE_mean':maxTE_var_m,   #nLoc x nLoc  
               'maxTE_sdev':maxTE_var_s,   #nLoc x nLoc  
               'maxMI_mean':maxMI_var_m,   #nLoc x nLoc  
               'maxMI_sdev':maxMI_var_s,   #nLoc x nLoc  
               'corrCoef_mean':corrCoef_var_m,  #nLoc x nLoc
               'corrCoef_sdev':corrCoef_var_s   #nLoc x nLoc
               }

if 'delayList' in out_VAR.keys():        
   postOut_var.update({'delayList':delayList,
                      'embDelay':embDelay_frcst, #nLoc x nRep   #embedding delays for computing TE
                     }) 
   if nDelay>1:        
      postOut_var.update({'AMIF_mean':AMIF_var_m,
                          'AMIF_sdev':AMIF_var_s})  

if not os.path.exists(outDir):
   os.makedirs(outDir)

with open(outDir+'postOut_sim_interacts','wb') as F:
   pickle.dump(postOut_sim,F)  

with open(outDir+'postOut_var_interacts','wb') as F:
   pickle.dump(postOut_var,F)         

print('Simulation out:',postOut_sim['it_opts'])
print('VAR out:',postOut_var['it_opts'],postOut_var['frcst_opts'])

print('------------------------')
print('Succuessfully written:')
print(f'    {outDir}postOut_sim_interacts')
print(f'    {outDir}postOut_var_interacts')
#
