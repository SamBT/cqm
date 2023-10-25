#from imports import *
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
import re
from sklearn.preprocessing import RobustScaler, QuantileTransformer, MinMaxScaler

#sys.path.append("/home/sambt/CASE_final/CASE_QUAK_final/")
from nflows.flows.base import Flow
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseQuadraticAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation

from customFlows import IndependentRQS, AutoregressiveRQS, Conditional1DRQS, ConditionalMultiRQS

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.utils.data as utils
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else print ('cpu')

class InfiniteLoader(utils.DataLoader):
    """A data loader that can load a dataset repeatedly."""

    def __init__(self, num_epochs=None, *args, **kwargs):
        """Constructor.

        Args:
            dataset: A `Dataset` object to be loaded.
            batch_size: int, the size of each batch.
            shuffle: bool, whether to shuffle the dataset after each epoch.
            drop_last: bool, whether to drop last batch if its size is less than
                `batch_size`.
            num_epochs: int or None, number of epochs to iterate over the dataset.
                If None, defaults to infinity.
        """
        super().__init__(
            *args, **kwargs
        )
        self.finite_iterable = super().__iter__()
        self.counter = 0
        self.num_epochs = float('inf') if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = super().__iter__()
            return next(self.finite_iterable)

    def __iter__(self):
        return self

    def __len__(self):
        return None

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers,act=None,out_act=None,dropout=0.0,bn=False):
        super().__init__()
        layers = []
        dcurr = input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(dcurr, hidden_dim))
            layers.append(nn.ReLU() if act is None else act)
            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            dcurr = hidden_dim
        layers.append(nn.Linear(dcurr, output_dim))
        if out_act is not None:
            self.out_act = out_act
            self.do_out = True
        else:
            self.do_out = False
        self.net = nn.Sequential(*layers)
        self.batchnorm=nn.BatchNorm1d(output_dim)
        self.bn = bn

    def forward(self, x):
        x = self.net(x)
        if self.bn:
            x = self.batchnorm(x)
        if self.do_out:
            x = self.out_act(x)
        return x

loss_dict = {
    "MSE":nn.MSELoss(),
    "BCE":nn.BCELoss()
}
    
def train_nn(net,inputs,labels,val_inputs,val_labels,n_epoch=20,lr=1e-3,bs=256,lossfunc="BCE",plot=True,losses=False):
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=lr)
    dataset = utils.TensorDataset(torch.tensor(inputs,dtype=torch.float32,device=device),torch.tensor(labels,dtype=torch.float32,device=device))
    val_dataset = utils.TensorDataset(torch.tensor(val_inputs,dtype=torch.float32,device=device),torch.tensor(val_labels,dtype=torch.float32,device=device))
    loader = InfiniteLoader(dataset=dataset,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
    val_loader = utils.DataLoader(dataset=val_dataset,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
    n_iter = n_epoch*inputs.shape[0]//bs
    n_iter_val = inputs.shape[0]//(bs)
    loss_fn = loss_dict[lossfunc]
    losses = []
    val_losses = []
    tbar = tqdm(range(n_iter))
    best_state = None
    min_loss = 1e+8
    patience_count = 0
    l_epoch = []
    for i in tbar:
        x,y = next(loader)
        optimizer.zero_grad()
        loss = loss_fn(net(x),y)
        loss.backward()
        optimizer.step()
        #losses.append(loss.item())
        l_epoch.append(loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            patience_count=0
            best_state = net.state_dict()
        else:
            patience_count+=1
        s = 'Loss: {0}, p = {1}'.format(loss.item(),patience_count)
        tbar.set_description(s)
        if (i+1)%n_iter_val == 0:
            losses.append(np.mean(l_epoch))
            l_epoch = []
            vlosses = []
            with torch.no_grad():
                for i, (x,y) in enumerate(val_loader):
                    vlosses.append(loss_fn(net(x),y).item())
            val_losses.append(np.mean(vlosses))

    net.load_state_dict(best_state)
    net.eval()
    torch.cuda.empty_cache()
    net = net.to('cpu')
    
    if plot:
        plt.figure(figsize=(8,6))
        #w = int(inputs.shape[0]/(bs))
        #smooth = np.convolve(np.ones(w),losses,mode='valid')/w
        #xvals = np.linspace(0,len(losses),len(smooth))
        #plt.plot(xvals,smooth)
        plt.plot(np.arange(len(losses)),losses)
        xvals = np.linspace(0,len(losses),len(val_losses))
        plt.plot(xvals,val_losses)
    if losses:
        return net, losses, val_losses
    else:
        return net

# numba for fast binning
from numba import vectorize, guvectorize, float64, int64, float32, int32, jit
@guvectorize([(float32[:,:],float32[:,:],int32[:,:])], '(n,m),(n,k)->(n,m)')
def nbinAll(XX,YY,res):
    N,M = XX.shape
    K = YY.shape[1]
    for i in range(N):
        X = XX[i]
        Y = YY[i]
        for j in range(M):
            x = X[j]
            if x < Y[0]:
                ind = 0
            elif x>=Y[-1]:
                ind = len(Y)
            else:
                for l in range(K-1):
                    if x >= Y[l] and x < Y[l+1]:
                        ind = l+1
                        break
            res[i,j] = ind

# numba for applying corrections
@guvectorize([(float32[:],float32[:],float32[:,:],float32[:,:],float32[:])], '(n),(k),(n,k),(n,k)->(n)')
def correctAll(Y,x,qbkg,qdata,Yout):
    N,M = qbkg.shape
    bad = 0
    bad_bkg = 0
    bad_data = 0
    underflow = 0
    overflow = 0
    good = 0
    for i in range(N):
        qb = qbkg[i]
        qd = qdata[i]
        y = Y[i]
        if y < x[0] or y > x[-1]:
            Yout[i] = y
        else:
            for j in range(M-1):
                if y >= x[j] and y < x[j+1]:
                    m = (qb[j+1]-qb[j])/(x[j+1]-x[j])
                    p = qb[j] + m*(y-x[j])
                    for k in range(M-1):
                        if p >= qd[k] and p < qd[k+1]:
                            ycorr = x[k] + (p-qd[k])*(x[k+1]-x[k])/(qd[k+1]-qd[k])
                    break
        Yout[i] = ycorr

class chainedNFTrainer:
    def __init__(self,projName,bkg_train,bkg_test,data_train,data_test,varNames,control=[],NF_kwargs={},outDir="cqm_models/",rangeScale=3,separateScale=False,minmax=True,useScale=True):
        assert len(varNames) == bkg_train.shape[1] and len(varNames) == data_train.shape[1]
        self.projName = projName
        self.varNames = varNames
        self.controlVars = [varNames[k] for k in control]
        self.control = control
        self.fitVars = [varNames[i] for i in range(len(varNames)) if i not in control]
        self.NF_kwargs = NF_kwargs
        self.outDir = f"{outDir}/{projName}/"
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        self.rangeScale = rangeScale
        self.separateScale = separateScale
        self.minmax = minmax
        self.useScale = useScale
        
        self.varDict = {v:v for v in varNames}
        
        # saving means, stds, mins, maxes for scaling
        bkg_all = np.concatenate((bkg_train,bkg_test),axis=0)
        self.bkg_maxes = bkg_all.max(axis=0)
        self.bkg_mins = bkg_all.min(axis=0)
        #self.bkg_maxes = np.quantile(bkg_all,0.9999,axis=0)
        #self.bkg_mins = np.quantile(bkg_all,0.0001,axis=0)
        self.bkg_means = bkg_all.mean(axis=0)
        self.bkg_stds = bkg_all.std(axis=0)
        data_all = np.concatenate((data_train,data_test),axis=0)
        self.data_maxes = data_all.max(axis=0)
        self.data_mins = data_all.min(axis=0)
        #self.data_maxes = np.quantile(data_all,0.9999,axis=0)
        #self.data_mins = np.quantile(data_all,0.0001,axis=0)
        self.data_means = data_all.mean(axis=0)
        self.data_stds = data_all.std(axis=0)
        
        self.writeInfo()
        
        # normalize data for trainings
        bkg_train = self.scale(bkg_train,"bkg",forward=True)
        bkg_test = self.scale(bkg_test,"bkg",forward=True)
        data_train = self.scale(data_train,"data",forward=True)
        data_test = self.scale(data_test,"data",forward=True)
        
        # set up dictionaries of train/test data
        self.data_train = {self.varNames[i]:data_train[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        self.data_test = {self.varNames[i]:data_test[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        self.bkg_train = {self.varNames[i]:bkg_train[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        self.bkg_test = {self.varNames[i]:bkg_test[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        
        # variables to track training -- keep *raw* values in the 'corrected' collection b/c data and MC have different preprocessing scaling
        self.correctedBkg_train = {self.varNames[i]:self.scale(bkg_train[:,i],"bkg",forward=False,idx=i).reshape(-1,1) for i in range(len(self.varNames))}
        self.correctedBkg_test = {self.varNames[i]:self.scale(bkg_test[:,i],"bkg",forward=False,idx=i).reshape(-1,1) for i in range(len(self.varNames))}
        self.data_models = [None for _ in range(len(self.fitVars))]
        self.bkg_models = [None for _ in range(len(self.fitVars))]
        self.data_model_locs = [None for _ in range(len(self.fitVars))]
        self.bkg_model_locs = [None for _ in range(len(self.fitVars))]
        self.current = 0 # index of variable currently being corrected (index in self.varOrder)
        self.bkg_trainings = {n:{} for n in self.fitVars}
        self.data_trainings = {n:{} for n in self.fitVars}
        self.bkg_base_dists = {n:None for n in self.fitVars}
        
    def writeInfo(self):
        # write basic info to json in base directory
        out_js = {}
        out_js['projName'] = self.projName
        out_js['varNames'] = self.varNames
        out_js['control'] = self.control
        out_js['controlVars'] = self.controlVars
        out_js['fitVars'] = self.fitVars
        out_js['baseDir'] = self.outDir
        out_js['rangeScale'] = self.rangeScale
        out_js['separateScale'] = self.separateScale
        out_js['minmax'] = self.minmax
        out_js['useScale'] = self.useScale
        out_js['bkg_maxes'] = self.bkg_maxes.tolist()
        out_js['bkg_mins'] = self.bkg_mins.tolist()
        out_js['bkg_means'] = self.bkg_means.tolist()
        out_js['bkg_stds'] = self.bkg_stds.tolist()
        out_js['data_maxes'] = self.data_maxes.tolist()
        out_js['data_mins'] = self.data_mins.tolist()
        out_js['data_means'] = self.data_means.tolist()
        out_js['data_stds'] = self.data_stds.tolist()
        with open(f"{self.outDir}/info.json","w") as info_out:
            json.dump(out_js,info_out,indent=4)
        
    @property
    def currentDir(self):
        currDir = f"{self.outDir}/step{self.current}_{self.fitVars[self.current]}/"
        if not os.path.isdir(currDir):
            os.makedirs(currDir)
        return currDir
        
    def scale(self,inputs,mode,forward=True,idx=None):
        if not self.useScale:
            return inputs
        else:
            if not self.separateScale:
                mins = np.minimum(self.bkg_mins,self.data_mins)
                maxes = np.maximum(self.bkg_maxes,self.data_maxes)
            else:
                mins = self.bkg_mins if mode=='bkg' else self.data_mins
                maxes = self.bkg_maxes if mode=='bkg' else self.data_maxes
            means = self.bkg_means
            stds = self.bkg_stds
            scale = np.where(np.abs(maxes)>np.abs(mins),np.abs(maxes),np.abs(mins))
            if idx is not None:
                mins,maxes,means,stds = mins[idx],maxes[idx],means[idx],stds[idx]
                scale = scale[idx]
            if forward:
                if self.minmax:
                    inputs = 2*self.rangeScale*((inputs-mins)/(maxes-mins)-0.5)
                else:
                    inputs = (inputs-means)/stds
                #inputs = self.rangeScale*(inputs/scale - means)
            else:
                if self.minmax:
                    inputs = (maxes-mins)*(inputs/(2*self.rangeScale) + 0.5) + mins
                else:
                    inputs = inputs*stds + means
                #inputs = (inputs/self.rangeScale + means)*scale
            return inputs
            
    def new_flow(self,num_features,num_context,kwargs,base_dist=None):
        return make_flow(num_features,num_context,kwargs,base_dist=base_dist)
    
    def get_flow(self,n_features,n_context,loc,kwargs,base_dist=None):
        flow = self.new_flow(n_features,n_context,kwargs,base_dist=base_dist)
        flow.load_state_dict(torch.load(loc))
        flow.eval()
        return flow
    
    def train_flow(self,flow,loader,name,kwargs,n_avg=100,anneal=True):
        flow = flow.to(device)
        if kwargs['wd']>0:
            optimizer = optim.Adam(flow.parameters(),lr=kwargs['learning_rate'],weight_decay=kwargs['wd'])
        else:
            optimizer = optim.Adam(flow.parameters(),lr=kwargs['learning_rate'])
        if anneal:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=kwargs['n_epoch'],eta_min=0.01*kwargs['learning_rate'],verbose=False)

        min_loss = 1e+8
        train_losses = []
        patience_count = 0
        saveName = f"{self.currentDir}/{name}.pt"
        
        print("Training flow {0}".format(name))
    
        tbar = tqdm(range(int(kwargs['n_epoch'])))
        avg_losses = []
        for i in tbar:
            if patience_count == kwargs['patience']:
                break
            epoch_losses = []
            x = next(loader)
            #for batch_idx, x in enumerate(loader):
            inputs,context = x
            optimizer.zero_grad()
            if torch.all(context==0):
                #loss = -flow.log_prob(inputs=inputs)[0].mean()
                loss = -flow.log_prob(inputs=inputs).mean()
            else:
                #loss = -flow.log_prob(inputs=inputs,context=context)[0].mean()
                loss = -flow.log_prob(inputs=inputs,context=context).mean()
            loss.backward()
            #gclip = 2
            #clip_grad_norm_(flow.parameters(), gclip)
            optimizer.step()
            train_losses.append(loss.item())
            if loss.item() < min_loss:
                min_loss = loss.item()
                patience_count = 0
                torch.save(flow.state_dict(),saveName)
            else:
                patience_count += 1
            avg_losses.append(loss.item())
            if i+1 > n_avg:
                avg_losses = avg_losses[1:]
            l_print = np.mean(avg_losses)
            s = 'Loss: {0}, p = {1}'.format(l_print,patience_count)
            tbar.set_description(s)
            if anneal:
                scheduler.step()
        
        flow.load_state_dict(torch.load(saveName))
        flow.eval()
        torch.cuda.empty_cache()
        flow = flow.to('cpu')

        return flow, saveName, train_losses
    
    def trainCurrentBkg(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,wd=0,anneal=True):
        train_kwargs = {'n_epoch':n_epoch,'patience':patience,'learning_rate':learning_rate,'wd':wd}
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        contextIdx = [self.varNames.index(v) for v in contextVars]
        
        # train bkg flow
        if len(contextVars) > 0:
            bkg_train_context = torch.tensor(self.scale(
                np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),
                "bkg",forward=True,idx=contextIdx),dtype=torch.float32,device=device)
        else:
            bkg_train_context = torch.zeros(self.bkg_train[currentVar].shape,dtype=torch.float32,device=device)
        bkg_train_var = torch.tensor(self.bkg_train[currentVar],dtype=torch.float32,device=device)
        bkg_train_dataset = utils.TensorDataset(bkg_train_var,bkg_train_context)
        bkg_loader = InfiniteLoader(dataset=bkg_train_dataset,batch_size=bs,shuffle=True,
                                    generator=torch.Generator(device='cuda'))
        bkg_flow = self.new_flow(1,len(contextVars),self.NF_kwargs)
        bkg_flowName = "bkgFlow_step{0}_{1}".format(self.current,self.fitVars[self.current])
        bkg_flow, bkg_flowLoc, bkg_trainLosses = self.train_flow(bkg_flow,bkg_loader,bkg_flowName,train_kwargs,anneal=anneal)
        self.bkg_trainings[currentVar]['flowLoc'] = bkg_flowLoc
        self.bkg_trainings[currentVar]['flowName'] = bkg_flowName
        self.bkg_trainings[currentVar]['contextVars'] = contextVars
        self.bkg_trainings[currentVar]['NF_kwargs'] = self.NF_kwargs
        self.bkg_trainings[currentVar]['train_kwargs'] = train_kwargs
        self.bkg_trainings[currentVar]['losses'] = bkg_trainLosses
        
        #encoder = deepcopy(bkg_flow._distribution._context_encoder.to('cpu'))
        #for param in encoder.parameters():
        #    param.requires_grad = False
        #self.bkg_base_dists[currentVar] = ConditionalDiagonalNormal(shape=[1],context_encoder=encoder)
        
        bkg_flowConfig = "bkgFlowConfig_step{0}_{1}".format(self.current,self.fitVars[self.current])
        with open(f"{self.currentDir}/{bkg_flowConfig}.json","w") as cfg_out:
            json.dump(self.bkg_trainings[currentVar],cfg_out,indent=4)
        
        plt.figure(figsize=(8,6))
        w = int(self.bkg_train[currentVar].shape[0]/(5*bs))
        smooth = np.convolve(np.ones(w),bkg_trainLosses,mode='valid')/w
        xvals = np.linspace(0,len(bkg_trainLosses),len(smooth))
        plt.plot(xvals,smooth)
        #plt.plot(np.arange(len(bkg_trainLosses)),bkg_trainLosses)
        plt.title(bkg_flowName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{self.currentDir}/trainCurve_bkg.pdf")
        
        del bkg_train_context, bkg_train_var, bkg_train_dataset, bkg_loader, bkg_flow
        torch.cuda.empty_cache()
        
    def trainCurrentData(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,wd=0,anneal=True):
        train_kwargs = {'n_epoch':n_epoch,'patience':patience,'learning_rate':learning_rate,'wd':wd}
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        
        # train data flow
        if len(contextVars) > 0:
            data_train_context = torch.tensor(np.concatenate([self.data_train[n] for n in contextVars],axis=1),dtype=torch.float32,device=device)
        else:
            data_train_context = torch.zeros(self.data_train[currentVar].shape,dtype=torch.float32,device=device)
        data_train_var = torch.tensor(self.data_train[currentVar],dtype=torch.float32,device=device)
        data_train_dataset = utils.TensorDataset(data_train_var,data_train_context)
        data_loader = InfiniteLoader(dataset=data_train_dataset,batch_size=bs,shuffle=True,
                                     generator=torch.Generator(device='cuda'))
        data_flow = self.new_flow(1,len(contextVars),self.NF_kwargs,base_dist=self.bkg_base_dists[currentVar])
        data_flowName = "dataFlow_step{0}_{1}".format(self.current,self.fitVars[self.current])
        data_flow, data_flowLoc, data_trainLosses = self.train_flow(data_flow,data_loader,data_flowName,train_kwargs,anneal=anneal)
        self.data_trainings[currentVar]['flowLoc'] = data_flowLoc
        self.data_trainings[currentVar]['flowName'] = data_flowName
        self.data_trainings[currentVar]['contextVars'] = contextVars
        self.data_trainings[currentVar]['NF_kwargs'] = self.NF_kwargs
        self.data_trainings[currentVar]['train_kwargs'] = train_kwargs
        self.data_trainings[currentVar]['losses'] = data_trainLosses
        
        data_flowConfig = "dataFlowConfig_step{0}_{1}".format(self.current,self.fitVars[self.current])
        with open(f"{self.currentDir}/{data_flowConfig}.json","w") as cfg_out:
            json.dump(self.data_trainings[currentVar],cfg_out,indent=4)
        
        plt.figure(figsize=(8,6))
        w = int(self.data_train[currentVar].shape[0]/(5*bs))
        smooth = np.convolve(np.ones(w),data_trainLosses,mode='valid')/w
        xvals = np.linspace(0,len(data_trainLosses),len(smooth))
        plt.plot(xvals,smooth)
        #plt.plot(np.arange(len(data_trainLosses)),data_trainLosses)
        plt.title(data_flowName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{self.currentDir}/trainCurve_data.pdf")
        del data_train_context, data_train_var, data_train_dataset, data_loader, data_flow
        torch.cuda.empty_cache()
        
    def trainCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,wd=0,anneal=True):
        self.trainCurrentBkg(bs=bs,n_epoch=n_epoch,patience=patience,
                              learning_rate=learning_rate,wd=wd,anneal=anneal)
        self.trainCurrentData(bs=bs,n_epoch=n_epoch,patience=patience,
                              learning_rate=learning_rate,wd=wd,anneal=anneal)
        self.plotDensity(bkg=True,data=False)
        self.plotDensity(bkg=False,data=True)
        
    def loadCurrent(self):
        currentVar = self.fitVars[self.current]
        with open(f"{self.currentDir}/dataFlowConfig_step{self.current}_{currentVar}.json","r") as f:
            self.data_trainings[currentVar] = json.load(f)
        with open(f"{self.currentDir}/bkgFlowConfig_step{self.current}_{currentVar}.json","r") as f:
            self.bkg_trainings[currentVar] = json.load(f)
        
    def plotDensity(self,bkg=False,data=False,ylim=None):
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        contextIdx = [self.varNames.index(v) for v in contextVars]
        
        if bkg:
            bkg_flow = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),
                                     self.bkg_trainings[currentVar]['flowLoc'],
                                     self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
            bkg_flow.eval()
            bkg_test_var = self.bkg_test[currentVar]
            nTest = bkg_test_var.shape[0]
            n_per = 20000
            split = np.array_split(np.arange(nTest),nTest//n_per + 1)
            samples = []
            if len(contextVars) > 0:
                bkg_test_context = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
                for k in split:
                    samp_k = torch.tensor(bkg_test_context[k],dtype=torch.float32,device=device)
                    with torch.no_grad():
                        samples.append(bkg_flow.sample(1,context=samp_k).detach().cpu().numpy().reshape(k.shape[0],-1))
                    del samp_k
                    torch.cuda.empty_cache()
                del bkg_test_context
            else:
                with torch.no_grad():
                    samples.append(bkg_flow.sample(bkg_test_var.shape[0]).detach().cpu().numpy())
            bkg_samples = np.concatenate(samples,axis=0)
            plotName = f"{self.currentDir}/bkg_fitDensity_step{self.current}_{self.fitVars[self.current]}.pdf"
            self.plotPair(bkg_test_var[:,0],bkg_samples[:,0],'bkg',plotName,log=False,ylim=ylim)
            plotName = f"{self.currentDir}/bkg_fitDensity_step{self.current}_{self.fitVars[self.current]}_log.pdf"
            self.plotPair(bkg_test_var[:,0],bkg_samples[:,0],'bkg',plotName,log=True,ylim=ylim)
            del bkg_flow, bkg_samples, bkg_test_var
            torch.cuda.empty_cache()
        
        if data:
            base_dist = self.bkg_base_dists[currentVar]
            data_flow = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs'],base_dist=base_dist).to(device)
            data_flow.eval()
            data_test_var = self.data_test[currentVar]
            nTest = data_test_var.shape[0]
            n_per = 20000
            split = np.array_split(np.arange(nTest),nTest//n_per + 1)
            samples = []
            if len(contextVars) > 0:
                data_test_context = np.concatenate([self.data_test[n] for n in contextVars],axis=1)
                for k in split:
                    samp_k = torch.tensor(data_test_context[k],dtype=torch.float32,device=device)
                    samples.append(data_flow.sample(1,context=samp_k).detach().cpu().numpy().reshape(k.shape[0],-1))
                    del samp_k
                    torch.cuda.empty_cache()
                del data_test_context
            else:
                samples.append(data_flow.sample(data_test_var.shape[0]).detach().cpu().numpy())
            data_samples = np.concatenate(samples,axis=0)
            plotName = f"{self.currentDir}/data_fitDensity_step{self.current}_{self.fitVars[self.current]}.pdf"
            self.plotPair(data_test_var[:,0],data_samples[:,0],'data',plotName,log=False,ylim=ylim)
            plotName = f"{self.currentDir}/data_fitDensity_step{self.current}_{self.fitVars[self.current]}_log.pdf"
            self.plotPair(data_test_var[:,0],data_samples[:,0],'data',plotName,log=True,ylim=ylim)
            del data_flow, data_samples, data_test_var
            torch.cuda.empty_cache()
            
    def plotPair(self,ref,samples,sampName,saveName,log=False,ylim=None):
        plt.subplots(figsize=(8,6),nrows=2,ncols=1,gridspec_kw={'height_ratios':[3,1]},sharex=True)
        plt.subplot(211)
        h1,bins,_ = plt.hist(ref,bins=np.linspace(-self.rangeScale,self.rangeScale,50),
                             density=True,histtype='step',label=f'{sampName} test')
        h2,bins,_ = plt.hist(samples,bins=np.linspace(-self.rangeScale,self.rangeScale,50),
                            density=True,histtype='step',label=f'{sampName} samples')
        if log:
            plt.yscale('log')
        plt.legend()
        plt.subplot(212)
        w = bins[1]-bins[0]
        c = (bins[1:]+bins[:-1])/2
        h = np.divide(h2,h1,where=h1>0)
        plt.bar(x=c,height=h,width=w,align='center')
        if ylim is not None:
            plt.ylim(ylim)
            plt.yticks(np.linspace(ylim[0],ylim[1],5))
        else:
            plt.ylim([0.8,1.2])
            plt.yticks(np.arange(0.8,1.3,0.1))
        plt.grid(axis='y')
        plt.ylabel("Ratio")
        plt.xlabel(self.fitVars[self.current])
        plt.savefig(saveName)
        
    def correctCurrent(self,n_per=10000,pad=0.1):
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        contextIdx = [self.varNames.index(v) for v in contextVars]
        
        bkg_flow = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
        data_flow = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)
        
        
        # correct training set
        if len(contextVars) > 0:
            bkg_train_context_bkg = self.scale(np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
            bkg_train_context_data = self.scale(np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
        else:
            bkg_train_context_bkg = np.zeros(self.bkg_train[currentVar].shape)
            bkg_train_context_data = np.zeros(self.bkg_train[currentVar].shape)
        bkg_train_var = self.bkg_train[currentVar]
        nTrain = bkg_train_context_bkg.shape[0]
        split = np.array_split(np.arange(nTrain),nTrain//n_per + 1)
        bkg_train_context_bkg = [bkg_train_context_bkg[k] for k in split]
        bkg_train_context_data = [bkg_train_context_data[k] for k in split]
        bkg_train_var = [bkg_train_var[k] for k in split]
        bkg_train_corr = []
        for i in tqdm(range(len(split))):
            inputs = torch.tensor(bkg_train_var[i],dtype=torch.float32,device=device)
            context_bkg = torch.tensor(bkg_train_context_bkg[i],dtype=torch.float32,device=device)
            context_data = torch.tensor(bkg_train_context_data[i],dtype=torch.float32,device=device)
            if torch.all(context_bkg==0):
                noise = bkg_flow.transform_to_noise(inputs=inputs)
                corrected = data_flow._transform.inverse(noise)[0]
            else:
                noise = bkg_flow.transform_to_noise(inputs=inputs,context=context_bkg)
                if type(bkg_flow._distribution) == ConditionalDiagonalNormal:
                    mean_bkg,log_std_bkg = bkg_flow._distribution._compute_params(context_bkg)
                    mean_data,log_std_data = data_flow._distribution._compute_params(context_bkg)
                    std_bkg = torch.exp(log_std_bkg)
                    std_data = torch.exp(log_std_data)
                    noise = (noise-mean_bkg)/std_bkg
                    noise = std_data*noise + mean_data
                corrected = data_flow._transform.inverse(noise,context=context_data)[0]
            bkg_train_corr.append(corrected.detach().cpu().numpy())
            del inputs,context_bkg,context_data,noise,corrected
            torch.cuda.empty_cache()
        bkg_train_corr = self.scale(np.concatenate(bkg_train_corr,axis=0),"data",forward=False,idx=self.varNames.index(currentVar))
        self.correctedBkg_train[currentVar] = np.copy(bkg_train_corr)
        del bkg_train_corr
        
        # correct test set
        if len(contextVars) > 0:
            bkg_test_context_bkg = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
            bkg_test_context_data = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
        else:
            bkg_test_context_bkg = np.zeros(self.bkg_test[currentVar].shape)
            bkg_test_context_data = np.zeros(self.bkg_test[currentVar].shape)
        bkg_test_var = self.bkg_test[currentVar]
        nTest = bkg_test_context_bkg.shape[0]
        split = np.array_split(np.arange(nTest),nTest//n_per + 1)
        bkg_test_context_bkg = [bkg_test_context_bkg[k] for k in split]
        bkg_test_context_data = [bkg_test_context_data[k] for k in split]
        bkg_test_var = [bkg_test_var[k] for k in split]
        bkg_test_corr = []
        for i in tqdm(range(len(split))):
            inputs = torch.tensor(bkg_test_var[i],dtype=torch.float32,device=device)
            context_bkg = torch.tensor(bkg_test_context_bkg[i],dtype=torch.float32,device=device)
            context_data = torch.tensor(bkg_test_context_data[i],dtype=torch.float32,device=device)
            if torch.all(context_bkg==0):
                noise = bkg_flow.transform_to_noise(inputs=inputs)
                corrected = data_flow._transform.inverse(noise)[0]
            else:
                noise = bkg_flow.transform_to_noise(inputs=inputs,context=context_bkg)
                if type(bkg_flow._distribution) == ConditionalDiagonalNormal:
                    mean_bkg,log_std_bkg = bkg_flow._distribution._compute_params(context_bkg)
                    mean_data,log_std_data = data_flow._distribution._compute_params(context_bkg)
                    std_bkg = torch.exp(log_std_bkg)
                    std_data = torch.exp(log_std_data)
                    noise = (noise-mean_bkg)/std_bkg
                    noise = std_data*noise + mean_data
                corrected = data_flow._transform.inverse(noise,context=context_data)[0]
            bkg_test_corr.append(corrected.detach().cpu().numpy())
            del inputs,context_bkg,context_data,noise,corrected
            torch.cuda.empty_cache()
        bkg_test_corr = self.scale(np.concatenate(bkg_test_corr,axis=0),"data",forward=False,idx=self.varNames.index(currentVar))
        self.correctedBkg_test[currentVar] = np.copy(bkg_test_corr)
        del bkg_test_corr
        
        del bkg_flow, data_flow
        torch.cuda.empty_cache()
        
    def correctFull(self,bkg,n_per=10000):
        bkg_corr = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        bkg = self.scale(bkg,"bkg",forward=True)
        bkg = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        
        for j,v in enumerate(self.fitVars):
            currentVar = v
            contextVars = self.controlVars+self.fitVars[:j]
            contextIdx = [self.varNames.index(v) for v in contextVars]
            
            bkg_flow = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
            base_dist = self.bkg_base_dists[currentVar]
            data_flow = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs'],base_dist=base_dist).to(device)

            # correct training set
            if len(contextVars) > 0:
                context_bkg = self.scale(np.concatenate([bkg_corr[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
                context_data = self.scale(np.concatenate([bkg_corr[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
            else:
                context_bkg = np.zeros(bkg[currentVar].shape)
                context_data = np.zeros(bkg[currentVar].shape)
            inputs = bkg[currentVar]
            nEvts = inputs.shape[0]
            split = np.array_split(np.arange(nEvts),nEvts//n_per + 1)
            context_bkg = [context_bkg[k] for k in split]
            context_data = [context_data[k] for k in split]
            inputs = [inputs[k] for k in split]
            corr = []
            for i in tqdm(range(len(split))):
                inputs_i = torch.tensor(inputs[i],dtype=torch.float32,device=device)
                context_i_bkg = torch.tensor(context_bkg[i],dtype=torch.float32,device=device)
                context_i_data = torch.tensor(context_data[i],dtype=torch.float32,device=device)
                with torch.no_grad():
                    if torch.all(context_i_bkg==0):
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i)
                        corrected = data_flow._transform.inverse(noise)[0]
                    else:
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i,context=context_i_bkg)
                        corrected = data_flow._transform.inverse(noise,context=context_i_data)[0]
                corr.append(corrected.detach().cpu().numpy())
                del inputs_i,context_i_bkg,context_i_data,noise,corrected
                torch.cuda.empty_cache()
            corr = self.scale(np.concatenate(corr,axis=0),"data",forward=False,idx=self.varNames.index(currentVar))
            bkg_corr[currentVar] = np.copy(corr)
            del corr
        
        return bkg_corr
    
    def plotTriplet(self,ref,corr,uncorr,bins=50,xlabel="",title="",saveName="",log=False,xlim=None,ylim=None):
        plt.subplots(figsize=(8,6),nrows=2,ncols=1,gridspec_kw={'height_ratios':[3,1]},sharex=True)
        plt.subplot(211)
        if xlim is not None:
            bins = np.linspace(xlim[0],xlim[1],bins)
        h1,bins,_ = plt.hist(ref,bins=bins,histtype='step',label="Data",density=True,color='gray',fill=True,alpha=0.5)
        h2,bins,_ = plt.hist(corr,bins=bins,histtype='step',label="Corr Bkg",density=True,color='C0',linewidth=2)
        h3,bins,_ = plt.hist(uncorr,bins=bins,histtype='step',label='Uncorr Bkg',density=True,color='red',linewidth=2)
        plt.title(title)
        if log:
            plt.yscale('log')
        plt.legend(loc='best')
        plt.subplot(212)
        w = bins[1]-bins[0]
        c = (bins[1:]+bins[:-1])/2
        h = np.divide(h2,h1,where=h1>0)
        plt.bar(x=c,height=h,width=w,align='center',color="C0")
        h = np.divide(h3,h1,where=h1>0)
        plt.step(x=bins[:-1],y=h,where='post',color="red",linewidth=2)
        plt.ylim([0.8,1.2] if ylim is None else ylim)
        plt.yticks(np.arange(0.8,1.3,0.1) if ylim is None else np.linspace(ylim[0],ylim[1],5))
        plt.grid(axis='y')
        plt.ylabel("Ratio")
        plt.xlabel(xlabel)
        plt.savefig(saveName)
    
    def plotVar(self,var,bins,xlim=None,ylim=None):
        idx = self.varNames.index(var)
        data_train = self.scale(self.data_train[var],"data",forward=False,idx=idx)
        data_test = self.scale(self.data_test[var],"data",forward=False,idx=idx)
        bkg_train = self.scale(self.bkg_train[var],"bkg",forward=False,idx=idx)
        bkg_test = self.scale(self.bkg_test[var],"bkg",forward=False,idx=idx)
        #corrBkg_train = self.scale(self.correctedBkg_train[var],"data",forward=False,idx=idx)
        #corrBkg_test = self.scale(self.correctedBkg_test[var],"data",forward=False,idx=idx)
        corrBkg_train = self.correctedBkg_train[var]
        corrBkg_test = self.correctedBkg_test[var]
        
        saveName = f"{self.outDir}/{var}_trainSet_beforeAfter.pdf"
        self.plotTriplet(data_train,corrBkg_train,bkg_train,bins,self.varDict[var],"Train Set",saveName,xlim=xlim,ylim=ylim)
        saveName = f"{self.outDir}/{var}_trainSet_beforeAfter_log.pdf"
        self.plotTriplet(data_train,corrBkg_train,bkg_train,bins,self.varDict[var],"Train Set",saveName,log=True,xlim=xlim,ylim=ylim)
        
        saveName = f"{self.outDir}/{var}_testSet_beforeAfter.pdf"
        self.plotTriplet(data_test,corrBkg_test,bkg_test,bins,self.varDict[var],"Test Set",saveName,xlim=xlim,ylim=ylim)
        saveName = f"{self.outDir}/{var}_testSet_beforeAfter_log.pdf"
        self.plotTriplet(data_test,corrBkg_test,bkg_test,bins,self.varDict[var],"Test Set",saveName,log=True,xlim=xlim,ylim=ylim)
        
    def plotCurrent(self,bins,xlim=None,ylim=None):
        currentVar = self.fitVars[self.current]
        self.plotVar(currentVar,bins,xlim=xlim,ylim=ylim)
        
    def plotAll(self,bins):
        for i,v in enumerate(self.controlVars+self.fitVars):
            b = bins[i] if type(bins)==list else bins
            self.plotVar(v,b)
        
    def runCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,bins=50,wd=0,anneal=True):
        print("RUNNING TRAININGS")
        self.trainCurrent(bs=bs,n_epoch=n_epoch,patience=patience,learning_rate=learning_rate,wd=wd,anneal=anneal)
        print("CORRECTING BKG")
        self.correctCurrent(n_per=10000)
        print("PLOTTING CORRECTIONS")
        self.plotCurrent(bins)
    
    def runAll(self,n_epoch=100,patience=20,learning_rate=1e-3,num_pt=100,bs=10000,wd=0):
        for i in range(len(self.fitVars)):
            nep = n_epoch[i] if type(n_epoch)==list else n_epoch
            pat = patience[i] if type(patience)==list else patience
            lr = learning_rate[i] if type(learning_rate)==list else learning_rate
            print("RUNNING TRAININGS")
            self.trainCurrent(bs=bs,n_epoch=nep,patience=pat,learning_rate=lr,wd=wd)
            print("CORRECTING BKG")
            self.correctCurrent(n_per=10000)
            self.stepForward()
        
    def stepForward(self):
        self.current += 1
        
    def stepBack(self):
        self.current -= 1
        
    def stepTo(self,step):
        self.current = step
        
class chainedNFCorrector:
    def __init__(self,baseDir,separateScale=False):
        self.baseDir = baseDir
        self.separateScale = separateScale
        self.dirs = [f"{baseDir}/{d}" for d in os.listdir(baseDir) if os.path.isdir(f"{baseDir}/{d}") and 'step' in d]
        self.dirs.sort(key=lambda x: int(re.search('step(\d+)',x).group(1)))
        self.steps = [f.split("/")[-1] for f in self.dirs]
        with open(f"{self.baseDir}/info.json") as f:
            self.info = json.load(f)
        for k in self.info.keys():
            setattr(self,k,self.info[k])
        self.bkg_means = np.array(self.bkg_means)
        self.bkg_stds = np.array(self.bkg_stds)
        self.bkg_mins = np.array(self.bkg_mins)
        self.bkg_maxes = np.array(self.bkg_maxes)
        self.data_means = np.array(self.data_means)
        self.data_stds = np.array(self.data_stds)
        self.data_mins = np.array(self.data_mins)
        self.data_maxes = np.array(self.data_maxes)
            
    def scale(self,inputs,mode,forward=True,idx=None):
        if not self.separateScale:
            mins = np.minimum(self.bkg_mins,self.data_mins)
            maxes = np.maximum(self.bkg_maxes,self.data_maxes)
        else:
            mins = self.bkg_mins if mode=='bkg' else self.data_mins
            maxes = self.bkg_maxes if mode=='bkg' else self.data_maxes
        means = self.bkg_means
        stds = self.bkg_stds
        scale = np.where(np.abs(maxes)>np.abs(mins),np.abs(maxes),np.abs(mins))
        if idx is not None:
            mins,maxes,means,stds = mins[idx],maxes[idx],means[idx],stds[idx]
            scale = scale[idx]
        if forward:
            inputs = 2*self.rangeScale*((inputs-mins)/(maxes-mins)-0.5)
            #inputs = (inputs-means)/stds
            #inputs = self.rangeScale*(inputs/scale - means)
        else:
            inputs = (maxes-mins)*(inputs/(2*self.rangeScale) + 0.5) + mins            
            #inputs = inputs*stds + means
            #inputs = (inputs/self.rangeScale + means)*scale
        return inputs
    
    def new_flow(self,num_features,num_context,kwargs):
        return make_flow(num_features,num_context,kwargs)
    
    def get_flow(self,n_features,n_context,loc,kwargs):
        flow = self.new_flow(n_features,n_context,kwargs)
        flow.load_state_dict(torch.load(loc))
        flow.eval()
        return flow
        
    def correctFull(self,bkg,n_per=10000):
        bkg_corr = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        bkg = self.scale(bkg,"bkg",forward=True)
        bkg = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        
        for j,v in enumerate(self.fitVars):
            currentVar = v
            contextVars = self.controlVars+self.fitVars[:j]
            contextIdx = [self.varNames.index(v) for v in contextVars]
            print(contextIdx)
            
            flowDir = f"{self.baseDir}/{self.dirs[j]}/"
            with open(f"{self.dirs[j]}/dataFlowConfig_{self.steps[j]}.json") as fj:
                dataFlowJson = json.load(fj)
            dataFlowLoc = f"{self.dirs[j]}/dataFlow_{self.steps[j]}.pt"
            with open(f"{self.dirs[j]}/bkgFlowConfig_{self.steps[j]}.json") as fj:
                bkgFlowJson = json.load(fj)
            bkgFlowLoc = f"{self.dirs[j]}/bkgFlow_{self.steps[j]}.pt"
            
            bkg_flow = self.get_flow(1,len(bkgFlowJson['contextVars']),bkgFlowLoc,bkgFlowJson['NF_kwargs']).to(device)
            data_flow = self.get_flow(1,len(dataFlowJson['contextVars']),dataFlowLoc,dataFlowJson['NF_kwargs']).to(device)

            # correct training set
            if len(contextVars) > 0:
                context_bkg = self.scale(np.concatenate([bkg_corr[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
                context_data = self.scale(np.concatenate([bkg_corr[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
            else:
                context_bkg = np.zeros(bkg[currentVar].shape)
                context_data = np.zeros(bkg[currentVar].shape)
            inputs = bkg[currentVar]
            nEvts = inputs.shape[0]
            split = np.array_split(np.arange(nEvts),nEvts//n_per + 1)
            context_bkg = [context_bkg[k] for k in split]
            context_data = [context_data[k] for k in split]
            inputs = [inputs[k] for k in split]
            corr = []
            for i in tqdm(range(len(split))):
                inputs_i = torch.tensor(inputs[i],dtype=torch.float32,device=device)
                context_i_bkg = torch.tensor(context_bkg[i],dtype=torch.float32,device=device)
                context_i_data = torch.tensor(context_data[i],dtype=torch.float32,device=device)
                with torch.no_grad():
                    if torch.all(context_i_bkg==0):
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i)
                        corrected = data_flow._transform.inverse(noise)[0]
                    else:
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i,context=context_i_bkg)
                        corrected = data_flow._transform.inverse(noise,context=context_i_data)[0]
                corr.append(corrected.detach().cpu().numpy())
                del inputs_i,context_i_bkg,context_i_data,noise,corrected
                torch.cuda.empty_cache()
            corr = self.scale(np.concatenate(corr,axis=0),"data",forward=False,idx=self.varNames.index(currentVar))
            bkg_corr[currentVar] = np.copy(corr)
            del corr
        
        return bkg_corr
    
    def invertFull(self,data,n_per=10000):
        data_corr = {self.varNames[i]:data[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        data = self.scale(data,"data",forward=True)
        data = {self.varNames[i]:data[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        
        for j in range(len(self.fitVars)-1,-1,-1):
            v = self.fitVars[j]
            currentVar = v
            contextVars = self.controlVars+self.fitVars[:j]
            contextIdx = [self.varNames.index(v) for v in contextVars]
            print(contextIdx)
            
            flowDir = f"{self.baseDir}/{self.dirs[j]}/"
            with open(f"{self.dirs[j]}/dataFlowConfig_{self.steps[j]}.json") as fj:
                dataFlowJson = json.load(fj)
            dataFlowLoc = f"{self.dirs[j]}/dataFlow_{self.steps[j]}.pt"
            with open(f"{self.dirs[j]}/bkgFlowConfig_{self.steps[j]}.json") as fj:
                bkgFlowJson = json.load(fj)
            bkgFlowLoc = f"{self.dirs[j]}/bkgFlow_{self.steps[j]}.pt"
            
            bkg_flow = self.get_flow(1,len(bkgFlowJson['contextVars']),bkgFlowLoc,bkgFlowJson['NF_kwargs']).to(device)
            data_flow = self.get_flow(1,len(dataFlowJson['contextVars']),dataFlowLoc,dataFlowJson['NF_kwargs']).to(device)

            # correct training set
            if len(contextVars) > 0:
                context_bkg = self.scale(np.concatenate([data_corr[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
                context_data = self.scale(np.concatenate([data_corr[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
            else:
                context_bkg = np.zeros(data[currentVar].shape)
                context_data = np.zeros(data[currentVar].shape)
            inputs = data[currentVar]
            nEvts = inputs.shape[0]
            split = np.array_split(np.arange(nEvts),nEvts//n_per + 1)
            context_bkg = [context_bkg[k] for k in split]
            context_data = [context_data[k] for k in split]
            inputs = [inputs[k] for k in split]
            corr = []
            for i in tqdm(range(len(split))):
                inputs_i = torch.tensor(inputs[i],dtype=torch.float32,device=device)
                context_i_bkg = torch.tensor(context_bkg[i],dtype=torch.float32,device=device)
                context_i_data = torch.tensor(context_data[i],dtype=torch.float32,device=device)
                with torch.no_grad():
                    if torch.all(context_i_bkg==0):
                        noise = data_flow.transform_to_noise(inputs=inputs_i)
                        corrected = bkg_flow._transform.inverse(noise)[0]
                    else:
                        noise = data_flow.transform_to_noise(inputs=inputs_i,context=context_i_data)
                        corrected = bkg_flow._transform.inverse(noise,context=context_i_bkg)[0]
                corr.append(corrected.detach().cpu().numpy())
                del inputs_i,context_i_bkg,context_i_data,noise,corrected
                torch.cuda.empty_cache()
            corr = self.scale(np.concatenate(corr,axis=0),"bkg",forward=False,idx=self.varNames.index(currentVar))
            data_corr[currentVar] = np.copy(corr)
            del corr
        
        return data_corr
    
def plotTriplet(ref,corr,uncorr,bins=50,log=False,ylim=None):
    fig,axes = plt.subplots(figsize=(8,6),nrows=2,ncols=1,gridspec_kw={'height_ratios':[3,1]},sharex=True)
    plt.subplot(211)
    h1,bins,_ = plt.hist(ref,bins=bins,histtype='step',label="Data",density=True,color='gray',fill=True,alpha=0.5)
    h2,bins,_ = plt.hist(corr,bins=bins,histtype='step',label="Corr Bkg",density=True,color='C0',linewidth=2)
    h3,bins,_ = plt.hist(uncorr,bins=bins,histtype='step',label='Uncorr Bkg',density=True,color='red',linewidth=2)
    if log:
        plt.yscale('log')
    plt.legend(loc='best')
    plt.subplot(212)
    w = bins[1]-bins[0]
    c = (bins[1:]+bins[:-1])/2
    h = np.divide(h2,h1,where=h1>0)
    plt.bar(x=c,height=h,width=w,align='center',color="C0")
    h = np.divide(h3,h1,where=h1>0)
    plt.step(x=bins[:-1],y=h,where='post',color="red",linewidth=2)
    plt.grid(axis='y')
    plt.ylabel("Ratio")
    plt.ylim([0,2])

def triplet(ref,corr,uncorr,axes,names,colors,bins=50,xlim=None):
    plt.sca(axes[0])
    h1,bins,_ = plt.hist(ref,bins=bins,histtype='step',label=names[0],density=True,color=colors[0],fill=True,alpha=0.5)
    h3,bins,_ = plt.hist(uncorr,bins=bins,histtype='step',label=names[1],density=True,color=colors[1],linewidth=3)
    h2,bins,_ = plt.hist(corr,bins=bins,histtype='step',label=names[2],density=True,color=colors[2],linewidth=3)
    h1c,bins = np.histogram(ref,bins=bins)
    h3c,bins = np.histogram(uncorr,bins=bins)
    h2c,bins = np.histogram(corr,bins=bins)
    if xlim is not None:
        plt.xlim(xlim)
    plt.sca(axes[1])
    w = bins[1]-bins[0]
    c = (bins[1:]+bins[:-1])/2
    h = np.divide(h2,h1,where=h1>0)
    herr = np.abs(h)*np.sqrt(np.divide(1.0,h2c.astype(np.float32),out=np.zeros_like(h2c.astype(np.float32)),where=h2c>0) + np.divide(1.0,h1c.astype(np.float32),out=np.zeros_like(h1c.astype(np.float32)),where=h1c>0))
    plt.errorbar(c,h,yerr=herr,color=colors[2],linewidth=0,markersize=15,marker='.',elinewidth=2,capsize=3,capthick=2)
    #h = list(np.divide(h2,h1,where=h1>0))
    #h.append(h[-1])
    #plt.step(x=bins,y=h,where='post',color=colors[2],linewidth=2)
    h = np.divide(h3,h1,where=h1>0)
    herr = np.abs(h)*np.sqrt(np.divide(1.0,h3c.astype(np.float32),out=np.zeros_like(h3c.astype(np.float32)),where=h3c>0) + np.divide(1.0,h1c.astype(np.float32),out=np.zeros_like(h1c.astype(np.float32)),where=h1c>0))
    plt.errorbar(c,h,yerr=herr,color=colors[1],linewidth=0,markersize=15,marker='.',elinewidth=2,capsize=3,capthick=2)
    #h = list(np.divide(h3,h1,where=h1>0))
    #h.append(h[-1])
    #plt.step(x=bins,y=h,where='post',color=colors[1],linewidth=2)
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylim([0,2])
    plt.yticks(np.arange(0,2.5,0.5))
    plt.grid(axis='y')
    
def plotPair(ref,samples,bins=50,saveName="",save=False,log=False,ylim=None):
    plt.subplots(figsize=(8,6),nrows=2,ncols=1,gridspec_kw={'height_ratios':[3,1]},sharex=True)
    plt.subplot(211)
    h1,bins,_ = plt.hist(ref,bins=bins,
                         density=True,histtype='step',label=f'reference')
    h2,bins,_ = plt.hist(samples,bins=bins,
                        density=True,histtype='step',label=f'samples')
    if log:
        plt.yscale('log')
    plt.legend()
    plt.subplot(212)
    w = bins[1]-bins[0]
    c = (bins[1:]+bins[:-1])/2
    h = np.divide(h2,h1,where=h1>0)
    plt.bar(x=c,height=h,width=w,align='center')
    if ylim is not None:
        plt.ylim(ylim)
        plt.yticks(np.linspace(ylim[0],ylim[1],5))
    else:
        plt.ylim([0.8,1.2])
        plt.yticks(np.arange(0.8,1.3,0.1))
    plt.grid(axis='y')
    plt.ylabel("Ratio")
    if save:
        plt.savefig(saveName)
        

def make_flow(num_features,num_context,kwargs,perm=False,base_dist=None):
    flow_type = kwargs['flow_type']
    if base_dist is None:
        if num_context == 0:
            base_dist = StandardNormal(shape=[num_features])
        else:
            #encoder = NeuralNet(num_context, 20, 2*num_features, 3, out_act=nn.Identity())
            encoder = nn.Linear(num_context,2*num_features)
            base_dist = ConditionalDiagonalNormal(shape=[num_features],context_encoder=encoder)
    base_dist = StandardNormal(shape=[num_features])
    transforms = []
    if num_context == 0:
        num_context = None
    for i in range(kwargs['num_layers']):
        if flow_type == 'MAF':
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features,
                                                                    hidden_features=kwargs['hidden_features'], 
                                                                    num_blocks=kwargs['num_blocks_per_layer']))
        elif flow_type == 'NSQUAD':
            transforms.append(MaskedPiecewiseQuadraticAutoregressiveTransform(features=num_features,
                                                                              context_features=num_context,
                                                                            hidden_features=num_features,
                                                                            num_bins=kwargs['num_bins'],
                                                                            num_blocks=kwargs['num_blocks_per_layer'],
                                                                            tail_bound=kwargs['tail_bound'], 
                                                                            tails='linear',
                                                                             dropout_probability=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                                                                use_batch_norm=kwargs['batchnorm'] if 'batchnorm' in kwargs.keys() else False))
        elif flow_type == 'NSRATQUAD':
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,
                                                                                      context_features=num_context,
                                                                                hidden_features=kwargs['hidden_features'],
                                                                                num_bins=kwargs['num_bins'],
                                                                                num_blocks=kwargs['num_blocks_per_layer'],
                                                                                tail_bound=kwargs['tail_bound'], 
                                                                                tails=kwargs['tails'],
                                                                                dropout_probability=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                                                                use_batch_norm=kwargs['batchnorm'] if 'batchnorm' in kwargs.keys() else False))
        elif flow_type == "IRQS":
            transforms.append(IndependentRQS(features=num_features,
                                             context=num_context,
                                             hidden=kwargs['hidden_features'],
                                             num_hidden=kwargs['num_blocks_per_layer'],
                                                num_bins=kwargs['num_bins'],
                                                tails=kwargs['tails'],
                                                tail_bound=kwargs['tail_bound'],
                                            dropout=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                            residual=kwargs['residual'] if 'residual' in kwargs.keys() else False))
        elif flow_type == "ARQS":
            transforms.append(AutoregressiveRQS(features=num_features,
                                             context=num_context,
                                             hidden=kwargs['hidden_features'],
                                             num_hidden=kwargs['num_blocks_per_layer'],
                                                num_bins=kwargs['num_bins'],
                                                tails=kwargs['tails'],
                                                tail_bound=kwargs['tail_bound'],
                                            dropout=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                            residual=kwargs['residual'] if 'residual' in kwargs.keys() else False))
        elif flow_type == "C1D":
            transforms.append(Conditional1DRQS(features=num_features,
                                             context=num_context,
                                             hidden=kwargs['hidden_features'],
                                             num_hidden=kwargs['num_blocks_per_layer'],
                                                num_bins=kwargs['num_bins'],
                                                tails=kwargs['tails'],
                                                tail_bound=kwargs['tail_bound'],
                                            dropout=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                            residual=kwargs['residual'] if 'residual' in kwargs.keys() else False))
        elif flow_type == "CMRQS":
            transforms.append(ConditionalMultiRQS(features=num_features,
                                                  num_context=num_context,
                                             hidden=kwargs['hidden_features'],
                                             num_hidden=kwargs['num_blocks_per_layer'],
                                                num_bins=kwargs['num_bins'],
                                                tails=kwargs['tails'],
                                                tail_bound=kwargs['tail_bound'],
                                            dropout=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                            residual=kwargs['residual'] if 'residual' in kwargs.keys() else False))
            
        
        if i < kwargs['num_layers'] - 1 and perm:
            transforms.append(ReversePermutation(features=num_features))
            #transforms.append(RandomPermutation(features=num_features))

    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

def train_flow(flow,data,context=None,n_epoch=20,lr=1e-3,bs=256,wd=0.0,patience=-1,n_avg=100,anneal=True):
    flow = flow.to(device)
    if wd>0:
        optimizer = optim.AdamW(flow.parameters(),lr=lr,weight_decay=wd)
    else:
        optimizer = optim.AdamW(flow.parameters(),lr=lr)
        
    n_iter = n_epoch*data.shape[0]//bs
    if anneal:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=n_iter,eta_min=lr,verbose=False)
    
    if context is None:
        dataset = utils.TensorDataset(torch.tensor(data,dtype=torch.float32,device=device))
    else:
        dataset = utils.TensorDataset(torch.tensor(data,dtype=torch.float32,device=device),torch.tensor(context,dtype=torch.float32,device=device))
    loader = InfiniteLoader(dataset=dataset,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))

    min_loss = 1e+8
    train_losses = []
    patience_count = 0
    bestState = None

    print("Training flow")

    tbar = tqdm(range(n_iter))
    avg_losses = []
    for i in tbar:
        if patience_count == patience:
            break
        optimizer.zero_grad()
        if context is None:
            x, = next(loader)
            #print(x.shape)
            #loss = -flow.log_prob(inputs=x)[0].mean()
            loss = -flow.log_prob(inputs=x).mean()
        else:
            x,y = next(loader)
            #loss = -flow.log_prob(inputs=x,context=y)[0].mean()
            loss = -flow.log_prob(inputs=x,context=y).mean()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            bestState = flow.state_dict()
            patience_count=0
        else:
            patience_count +=1
        avg_losses.append(loss.item())
        if i+1 > n_avg:
            avg_losses = avg_losses[1:]
        l_print = np.mean(avg_losses)
        s = 'Loss: {0}, p = {1}'.format(l_print,patience_count)
        tbar.set_description(s)
        if anneal:
            scheduler.step()

    flow.load_state_dict(bestState)
    flow.eval()
    del dataset, loader, optimizer
    #del scheduler
    torch.cuda.empty_cache()
    flow = flow.to('cpu')
    
    plt.figure(figsize=(8,6))
    w = int(data.shape[0]/(5*bs))
    smooth = np.convolve(np.ones(w),train_losses,mode='valid')/w
    xvals = np.linspace(0,len(train_losses),len(smooth))
    plt.plot(xvals,smooth)

    return flow