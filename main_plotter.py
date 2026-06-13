# Functions used to plot the post-processed data (used in the paper)
#
# https://github.com/simFlowDataLab/PairwiseValidation_turbulenceTS
# Saleh Rezaeiravesh, saleh.rezaeiravesh@manchester.ac.uk
#
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = False
rcParams['mathtext.fontset'] = 'cm'       
rcParams['font.family'] = 'sans-serif'         
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
#
def maxMeanSdevArray_find(A,B):
    """
    Find max(A) and B at argmax(A)
    A,B: nLoc x nLoc x lags
    max(A) is found over lags
    """
    Imax = np.argmax(A,axis=-1)
    nLoc = A.shape[0]
    M = np.zeros((nLoc,nLoc))
    S = np.zeros((nLoc,nLoc))
    for i in range(nLoc):
        for j in range(nLoc):
            M[i,j] = A[i,j,Imax[i,j]]
            S[i,j] = B[i,j,Imax[i,j]]
    return M,S,Imax     

def discreteMap_plot(f,ypls,iLocList,opts={}):
    """
    Plot discrete map of f with y+ as axes label
    """

    fig, ax = plt.subplots()
    im = ax.imshow(f,cmap='Spectral_r')

    cax = ax.inset_axes([0, 1.02, 1, 0.05], transform=ax.transAxes)

    vmin, vmax = im.get_clim()
    cbar = fig.colorbar(im,cax=cax,location='top')
    cbar.ax.tick_params(labelsize=13)
    cbar_ticks=np.linspace(vmin,vmax,8)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in cbar_ticks])

    labs = [str(round(ypls[i])) for i in range(len(iLocList))]
    labs[0] = str(round(ypls[0],1))

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(iLocList)), labels=labs,
                  rotation=90, va="center", ha='right', rotation_mode="anchor",fontsize=13)
    ax.set_yticks(range(len(iLocList)), labels=labs,fontsize=13)
    if 'xLab' in opts.keys():
        ax.set_xlabel(opts['xLab'],fontsize=20)
    else:
        ax.set_xlabel(r'$Target y^+$',fontsize=20)
    if 'yLab' in opts.keys():
        ax.set_ylabel(opts['yLab'],fontsize=20)
    else:
        ax.set_ylabel(r'$Source y^+$',fontsize=20)
    if 'title' in opts.keys():
       ax.set_title(opts['title'],fontsize=12)
    #save   
    if 'save' in opts.keys() and opts.keys():
       DPI = fig.get_dpi()
       fig.set_size_inches(600/float(DPI),500/float(DPI))
       figDir=opts['figDir']
       if not os.path.exists(figDir):
          os.makedirs(figDir)
       plt.savefig(figDir+opts['figName']+'.pdf',bbox_inches='tight')
    plt.show()
    return cbar

########################
####External Functions
########################
# ------ Setting---------------
# Directory where the pickle files of interactions are located
caseName = 'mpi_test'
# -----------------------------
#
inDir = './outData/'+caseName+'/postprcssd/'
#
with open(inDir+'postOut_sim_interacts','rb') as F:
    simData = pickle.load(F)
with open(inDir+'postOut_var_interacts','rb') as F:
    varData = pickle.load(F)          

#simulation-data interaction measures
ACF_sim = simData['ACF']    
CCF_sim = simData['CCF']    
TE_sim = simData['TE']    
MI_sim = simData['MI']   
corrCoef_sim = simData['corrCoef']    
maxTE_sim = simData['maxTE']    
maxMI_sim = simData['maxMI']    
if 'AMIF' in simData.keys():   #when optimising th embedding delay
   AMIF_sim = simData['AMIF']   

#normalized net maximum TE - simulation data
maxNetTE_sim = maxTE_sim - maxTE_sim.T  #max net TE simulation
maxNetTENrm_sim = np.where(maxNetTE_sim>=0.0, maxNetTE_sim, 0.0)  #only select net TE>0, rest =0
maxNetTENrm_sim /= np.max(maxNetTENrm_sim)  #max net TE simulation - normalized

embDelay_sim = simData['embDelay']

#VAR-data interaction measures
ACF_var_m = varData['ACF_mean']    
ACF_var_s = varData['ACF_sdev']    
CCF_var_m = varData['CCF_mean']    
CCF_var_s = varData['CCF_sdev']    
TE_var_m = varData['TE_mean']    
TE_var_s = varData['TE_sdev']    
MI_var_m = varData['MI_mean']    
MI_var_s = varData['MI_sdev']    
maxMI_var_m = varData['maxMI_mean']    
maxMI_var_s = varData['maxMI_sdev']    
maxTE_var_m = varData['maxTE_mean']    
maxTE_var_s = varData['maxTE_sdev']    
corrCoef_var_m = varData['corrCoef_mean']    
corrCoef_var_s = varData['corrCoef_sdev']    
if 'AMIF_mean' in varData.keys():  #onlty if the embedding delay has been optimised
   AMIF_var_m = varData['AMIF_mean']    
   AMIF_var_s = varData['AMIF_sdev']    
embDelay_var = varData['embDelay']

#normalized net maximum TE - VAR data
maxNetTE_var_m = maxTE_var_m - maxTE_var_m.T  #max net TE for VAR data
maxNetTENrm_var_m = np.where(maxNetTE_var_m>=0.0, maxNetTE_var_m, 0.0)  #only select net TE>0, rest =0
maxNetTENrm_var_m /= np.max(maxNetTENrm_var_m)  #max net TE foreacst - normalized

#common between simulation and VAR data
ncf = ACF_sim.shape[-1]
lagList = simData['lagList']
iLocList = simData['iLocList']
nLoc = len(iLocList)
yplsList=simData['yplsLocList']
delayList=simData['delayList']


print(simData.keys())
print(varData.keys())
print('n iLocList=',len(yplsList))
print(len(iLocList))
print('y + list:',np.round(np.array(yplsList)))
print('lagList',lagList)
#print(ACF_sim.shape)
print(TE_var_m.shape)
for i in range(len(iLocList)):
    print(f'i={i} , iLoc ={iLocList[i]} , y+ = {yplsList[i]}')

##############################################
## Used for Data Centric Engineering Paper
###############################################
#
def single_imshow():
    """
    Single imshow contour plot for a chosen quantity `M`
    axes: y+ source and target
    """
    ##---- settings
    #array to plot
    #M = corrCoef_var_s
    #M = maxMI_sim
    #M = (maxTE_var_m - maxTE_sim)/maxTE_sim
    #M = (maxMI_var_m - maxMI_sim)/maxMI_sim
    #M = maxTE_var_s/maxTE_var_m
    M = maxTE_var_m
    #M = maxMI_var_m
    #M = maxNetTENrm_sim
    #M = maxNetTENrm_var_m

    #axes labels 
    xLab = r'$\text{Target }y^+$'
    yLab = r'$\text{Source }y^+$'

    save = True
    figName='maxTE_var_m'+'_'+caseName
    ##---------------------------------
    #
    opts={'xLab':xLab,
          'yLab':yLab,  
          'save':save,
          'figDir':'./figsOut/'+caseName+'/',
          'figName':figName}

    discreteMap_plot(M,yplsList,iLocList,opts)
#
def single_scatterPlot():
    """
    Single scatter plot: synthetic vs. simulation measure
    """
    ##---- settings
    #array to plot
    sim = maxTE_sim   #simulation
    M = maxTE_var_m   #VAR mean
    S = maxTE_var_s   #VAR sdev

    #fig save
    save = True
    figName='maxTE_scatter'+'_'+caseName
    figDir='./figsOut/'+caseName+'/'
    ##---------------------------------
    #
    c_ = np.linspace(np.min(sim)*1.1,1.1*np.max(sim),100)
    fig = plt.figure(figsize=(5,5))
    m_ = M.flatten()
    s_ = 1.96*S.flatten()
    sim_ = sim.flatten()

    plt.errorbar(x=sim_,y=m_,yerr=s_,fmt='o',color='None',ecolor='steelblue',
                 markeredgecolor='k',alpha=0.4)
    plt.plot(c_,c_,'--',color='red')
    plt.xlabel(r'Simulation data',fontsize=14)
    plt.ylabel(r'Synthetic data',fontsize=14)
    plt.xlim([np.min(c_),np.max(c_)])
    plt.ylim([np.min(c_),np.max(c_)])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.grid()

    #save   
    if save:
       DPI = fig.get_dpi()
       fig.set_size_inches(500/float(DPI),500/float(DPI))
       if not os.path.exists(figDir):
          os.makedirs(figDir)
       plt.savefig(figDir+figName+'.pdf',bbox_inches='tight')
    plt.show()
#
def single_acf_ccf():
    """
    Plot ACF & CCF vresus lag for a pair of time series.  
    """
    ## --- setting
    #Index of the datasets to plot their ACF and CCF
    I1, I2 = 1, 2   

    #fig save
    save = True
    figName='cfs_'+str(I1)+'_'+str(I2)+'_'+caseName
    figDir='./figsOut/'+caseName+'/'
    ##----------------------------------------
    print(f'y+ values of the selected points: {yplsList[I1]}, {yplsList[I2]}')
    acf_lags = np.arange(ncf)

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    plt.subplot(221); ax=plt.gca();

    sim_ = ACF_sim[I1,I2,:]
    m_ = ACF_var_m[I1,I2,:]
    s_ = ACF_var_s[I1,I2,:]
    plt.semilogx(acf_lags,sim_,'--',color='indianred')
    plt.plot(acf_lags,m_,'-',color='steelblue')
    ax.fill_between(acf_lags, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    #plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$\rho_{xx}(\tau)$',fontsize=14)

    plt.subplot(222); ax=plt.gca();
    sim_ = ACF_sim[I2,I1,:]
    m_ = ACF_var_m[I2,I1,:]
    s_ = ACF_var_s[I2,I1,:]
    plt.semilogx(acf_lags,sim_,'--',color='indianred')
    plt.plot(acf_lags,m_,'-',color='steelblue')
    ax.fill_between(acf_lags, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    #plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$\rho_{yy}(\tau)$',fontsize=14)

    plt.subplot(223); ax=plt.gca();
    sim_ = CCF_sim[I1,I2,:]
    m_ = CCF_var_m[I1,I2,:]
    s_ = CCF_var_s[I1,I2,:]
    plt.semilogx(acf_lags,sim_,'--',color='indianred')
    plt.plot(acf_lags,m_,'-',color='steelblue')
    ax.fill_between(acf_lags, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$\gamma_{xy}(\tau)$',fontsize=14)

    plt.subplot(224); ax=plt.gca();
    sim_ = CCF_sim[I2,I1,:]
    m_ = CCF_var_m[I2,I1,:]
    s_ = CCF_var_s[I2,I1,:]
    plt.semilogx(acf_lags,sim_,'--',color='indianred')
    plt.plot(acf_lags,m_,'-',color='steelblue')
    ax.fill_between(acf_lags, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$\gamma_{yx}(\tau)$',fontsize=14)

    #save   
    if save:
       DPI = fig.get_dpi()
       fig.set_size_inches(500/float(DPI),400/float(DPI))
       if not os.path.exists(figDir):
          os.makedirs(figDir)
       plt.savefig(figDir+figName+'.pdf',bbox_inches='tight')

    plt.show()
#
def single_MI():
    """
    Plot MI (MIxy, MIyx) for a pair of time series
    """
    ## --- setting
    #Index of the datasets to plot their ACF and CCF 
    I1, I2 = 1, 2

    #fig save
    save = True
    figName='mi_'+str(I1)+'_'+str(I2)+'_'+caseName
    figDir='./figsOut/'+caseName+'/'
    ##----------------------------------------
    print(f'y+ values of the selected points: {yplsList[I1]}, {yplsList[I2]}')
    
    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    plt.subplot(121); ax=plt.gca();
    sim_ = MI_sim[I1,I2,:]
    m_ = MI_var_m[I1,I2,:]
    s_ = MI_var_s[I1,I2,:]
    plt.plot(lagList,sim_,'--',color='indianred')
    plt.plot(lagList,m_,'-',color='steelblue')
    ax.fill_between(lagList, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$I_{xy}(\tau)$',fontsize=14)

    plt.subplot(122); ax=plt.gca();
    sim_ = MI_sim[I2,I1,:]
    m_ = MI_var_m[I2,I1,:]
    s_ = MI_var_s[I2,I1,:]
    plt.plot(lagList,sim_,'--',color='indianred')
    plt.plot(lagList,m_,'-',color='steelblue')
    ax.fill_between(lagList, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$I_{yx}(\tau)$',fontsize=14)

    #save   
    if save:
       DPI = fig.get_dpi()
       fig.set_size_inches(500/float(DPI),200/float(DPI))
       if not os.path.exists(figDir):
          os.makedirs(figDir)
       plt.savefig(figDir+figName+'.pdf',bbox_inches='tight')

    plt.show()
#    
def single_AMIF():
    """
    Plot AMIF (MIxx, MIyy) for a pair of time series
    """
    ## --- setting
    #Index of the datasets to plot their ACF and CCF
    I1, I2 = 1, 2

    #fig save
    save = False
    figName='amif_'+str(I1)+'_'+str(I2)+'_'+caseName
    figDir='./figsOut/'+caseName+'/'
    ##----------------------------------------
    #
    print(f'y+ values of the selected points: {yplsList[I1]}, {yplsList[I2]}')
    
    fig, axes = plt.subplots(2, 1, constrained_layout=True)

    plt.subplot(211); ax=plt.gca();
    sim_ = AMIF_sim[I1,I2,:]
    m_ = AMIF_var_m[I1,I2,:]
    s_ = AMIF_var_s[I1,I2,:]
    jsim_ = np.where(sim_<sim_[0]/np.exp(1))[0][0]
    jvar_ = np.where(m_<m_[0]/np.exp(1))[0][0]
    print(jsim_,jvar_,delayList[jsim_])
    plt.plot(delayList,sim_,'--',color='indianred')
    plt.plot(delayList,m_,'-',color='steelblue')
    plt.axvline(delayList[jsim_],color='k',linestyle='dotted')
    #plt.axvline(jvar_)
    ax.fill_between(delayList, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    #plt.xlabel(r'$\tau$',fontsize=16)
    plt.ylabel(r'$I_{XX}(\tau)$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(212); ax=plt.gca();
    sim_ = AMIF_sim[I2,I1,:]
    m_ = AMIF_var_m[I2,I1,:]
    s_ = AMIF_var_s[I2,I1,:]
    jsim_ = np.where(sim_<sim_[0]/np.exp(1))[0][0]
    jvar_ = np.where(m_<m_[0]/np.exp(1))[0][0]
    print(jsim_,jvar_,delayList[jsim_])
    plt.plot(delayList,sim_,'--',color='indianred')
    plt.plot(delayList,m_,'-',color='steelblue')
    plt.axvline(delayList[jsim_],color='k',linestyle='dotted')
    ax.fill_between(delayList, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    plt.xlabel(r'$\tau$',fontsize=20)
    plt.ylabel(r'$I_{XX}(\tau)$',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    #save   
    if save:
       DPI = fig.get_dpi()
       fig.set_size_inches(500/float(DPI),500/float(DPI))
       if not os.path.exists(figDir):
          os.makedirs(figDir)
       plt.savefig(figDir+figName+'.pdf',bbox_inches='tight')

    plt.show()
#
def emb_delay():
    """
    Plot Optimal embedding delays for each y+
    """
    ## --- setting
    #fig save
    save = True
    figName='optimalTau_'+caseName
    figDir='./figsOut/'+caseName+'/'
    ##----------------------------------------
    #
    sim_ = embDelay_sim
    m_ = embDelay_var
   
    fig = plt.figure(figsize=(4,5))
    for i in range(m_.shape[1]): #number of repetition
        plt.plot(yplsList,m_[:,i],'x')
    plt.plot(yplsList,sim_,'ok',mfc='None',ms=10)

    plt.xlabel(r'$y^+$',fontsize=20)
    plt.ylabel(r'$\text{Optimal}\,\tau$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    #save   
    if save:
       DPI = fig.get_dpi()
       fig.set_size_inches(800/float(DPI),600/float(DPI))
       if not os.path.exists(figDir):
          os.makedirs(figDir)
       plt.savefig(figDir+figName+'.pdf',bbox_inches='tight')

    plt.show()

def single_TE():
    """
    Plot TE (TExy, TEyx) versus lag for a pair of time series
    """
    ## --- setting
    #Index of the datasets to plot their ACF and CCF
    I1, I2 = 1, 2

    #fig save
    save = False
    figName='te_'+str(I1)+'_'+str(I2)+'_'+caseName
    figDir='./figsOut/'+caseName+'/'
    ##----------------------------------------
    print(f'y+ values of the selected points: {yplsList[I1]}, {yplsList[I2]}')

    fig, axes = plt.subplots(2, 1, constrained_layout=True)
    leg_x = np.round(yplsList[I1])
    leg_y = np.round(yplsList[I2])

    plt.subplot(211); ax=plt.gca();
    sim_ = TE_sim[I1,I2,:]
    m_ = TE_var_m[I1,I2,:]
    s_ = TE_var_s[I1,I2,:]
    plt.title(r' %d $\to$ %d' %(leg_x,leg_y),fontsize=14)
    plt.plot(lagList,sim_,'--s',mfc='None',color='indianred',label='Simulation')
    plt.plot(lagList,m_,'-o',mfc='None',color='steelblue',label='Synthetic')
    ax.fill_between(lagList, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    #plt.xlabel(r'$m$',fontsize=14)
    if I1==1 and I2==3:
       plt.ylabel(r'${\rm TE}$',fontsize=16)
    plt.xticks(fontsize=14)
    xticks_ = np.arange(lagList[0],lagList[-1]+1,2)
    ax.set_xticks(xticks_)
    plt.yticks(fontsize=14)
    #plt.legend(loc='best',fontsize=14)

    plt.subplot(212); ax=plt.gca();
    sim_ = TE_sim[I2,I1,:]
    m_ = TE_var_m[I2,I1,:]
    s_ = TE_var_s[I2,I1,:]
    plt.title(r' %d $\to$ %d' %(leg_y,leg_x),fontsize=14)
    plt.plot(lagList,sim_,'--s',mfc='None',color='indianred')
    plt.plot(lagList,m_,'-o',mfc='None',color='steelblue')
    ax.fill_between(lagList, m_-1.96*s_,m_+1.96*s_,color='powderblue', alpha=0.5)
    plt.xlabel(r'$m$',fontsize=16)
    if I1==1 and I2==3:
       plt.ylabel(r'${\rm TE}$',fontsize=16)
    plt.xticks(fontsize=14)
    xticks_ = np.arange(lagList[0],lagList[-1]+1,2)
    ax.set_xticks(xticks_)
    plt.yticks(fontsize=14)

    #save   
    if save:
       DPI = fig.get_dpi()
       fig.set_size_inches(300/float(DPI),600/float(DPI))
       if not os.path.exists(figDir):
          os.makedirs(figDir)
       plt.savefig(figDir+figName+'.pdf',bbox_inches='tight')

    plt.show()

