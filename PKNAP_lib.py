
# import
import os,sys
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt

import sklearn.metrics as sm

import torch
import torch.nn as nn
import torch.nn.functional as F

rlist = list('ACDEFGHIKLMNPQRSTVWY')


# embedding
def Embedding_CLS(seqlist): # to be modified
  rdict = {}
  for i in range(len(rlist)):
    rdict[rlist[i]] = i
  data = np.array([[20] + [rdict[r] for r in s] for s in seqlist])
  return data

# model
class MyModule(nn.Module):
  def __init__(self, d_model=21, max_len=10, n_head=1, n_atlayer=1, dropout=0.0, d_ffn=4, device='cpu'):
    super().__init__()
    self.device = device
    self.max_len = max_len
    self.dm = d_model
    self.nh = n_head
    self.df = self.dm * d_ffn
    self.nat = n_atlayer
    self.dropout = dropout
    self.aa_emb = nn.Embedding(21, self.dm)
    self.pe = self.PositionalEncoding()
    self.texn = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model = self.dm,
                nhead = self.nh,
                dim_feedforward = self.df,
                dropout = self.dropout,
                batch_first = True),
                self.nat)
    self.linear_mm = nn.Linear(self.dm, self.dm)
    self.relu = nn.ReLU()
    self.linear_m6 = nn.Linear(self.dm, 6)
    self.linear_m1 = nn.Linear(self.dm, 1) # DEPRECATED but for version 2.3 or earlier

    # DEPRECATED
    #self.loss_pretrain = nn.BCEWithLogitsLoss()
    #self.loss_pretrain2 = nn.Hardtanh()

  def SaveParameters(self, opfn='model_parameters.pkl'):
    SP_dict = {'d_model':   self.dm, 
               'max_len':   self.max_len, 
               'n_head':    self.nh, 
               'n_atlayer': self.nat, 
               'dropout':   self.dropout, 
               }
    import pickle
    pickle.dump(SP_dict, open(opfn,'wb'))

  def ReportParameters(self):
    print ('Parameters')
    print (' d_model\t%s' %self.dm)
    print (' max_len\t%s' %self.max_len)
    print (' n_head \t%s' %self.nh)
    print (' n_atlayer\t%s' %self.nat)
    print (' dropout\t%s' %self.dropout)
    print (' device \t%s' %self.device)

  def PositionalEncoding(self):
    pe = torch.zeros(self.max_len, self.dm)
    position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
    i2 = torch.arange(0, self.dm, 2, dtype=torch.float)
    div_term = torch.exp( - i2 * np.log(10000) / self.dm )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(self.device)

  def BaseLayer(self, x):
    x = self.aa_emb(x) + self.pe
    # dropout?
    x = self.texn(x)
    # mask?
    return x

  def CLSBaseLayer(self, x):
    x = self.BaseLayer(x)
    x0 = x[:,0]
    x0 = self.linear_mm(x0)
    x0 = self.relu(x0)
    return x0

  def DGDecompLayer(self, x):
    x = self.CLSBaseLayer(x)
    x = self.linear_m6(x)
    return x.view(-1,2,3)

  def DGLayer(self, x):
    x = self.DGDecompLayer(x)
    x = x.sum(-1)
    x = x[:,1] - x[:,0]
    return x

  def ProbLayer(self, x):
    x = self.DGLayer(x)
    return torch.sigmoid(x)

  # DEPRECATED
  #def STDLayer(self, x):
  #  x = self.CLSBaseLayer(x)
  #  x = self.linear_m1(x)
  #  return x.squeeze()

  #def DGSTDLayer(self, x):
  #  x = self.CLSBaseLayer(x)
  #  dg = self.linear_m6(x)
  #  dg = dg.view(-1,2,3)
  #  dg = dg.sum(-1)
  #  dg = dg[:,1] - dg[:,0]
  #  se = self.linear_m1(x)
  #  se = se.squeeze()
  #  return dg, se

  #def KL_Gauss(self,u1,s12,u2,s22):
  #  return 0.5 * (torch.log(s22/s12) + (s12+(u1-u2)**2)/s22).mean() - 0.5

# eval
def Metric(model, Xyloader, precision=3):
  model.eval()
  yps, ys = [], []
  for X,y in Xyloader:
    #pred = model.DGLayer(X)
    #yp = torch.sigmoid(pred).cpu().detach().tolist()
    yp = model.ProbLayer(X).cpu().detach().tolist()
    yps = yps + yp
    ys = ys + y.cpu().detach().tolist()
  ys = np.array(ys)
  yps = np.array(yps)
  auc = np.round(sm.roc_auc_score(ys, yps),      precision)
  acc = np.round(sm.accuracy_score(ys, yps>0.5), precision)
  cel = np.round(sm.log_loss(ys, yps),           precision)
  return cel,acc,auc

# mutate
def Mutate(seq, Nmut):
  out = []
  labels = []
  for p in it.combinations(range(len(seq)), Nmut):
    for r in it.product(rlist, repeat=Nmut):
      seq_tmp = list(seq)
      label = ''
      for i in range(len(p)):
        r0i = seq_tmp[p[i]]
        ri = r[i]
        if r0i != ri:
          seq_tmp[p[i]] = ri
          label = label + r0i + str(p[i]+1) + ri + '+'
      out.append(''.join(seq_tmp))
      labels.append(label[:-1])
  return out, labels

def MutatePlot(arr, opfn=None):
  plt.figure(figsize=(6,3))
  plt.imshow(arr, cmap='bwr_r', vmin=-2, vmax=2)
  plt.colorbar(label='Predicted '+r'$\Delta\Delta$'+'G (kcal/mol)', ticks=np.arange(-2,3,1), fraction=0.05, shrink=0.83)
  plt.xlabel('Mutated Residue')
  plt.ylabel('Position')
  plt.xticks(np.arange(20), rlist)
  plt.yticks(np.arange(9), np.arange(1,10))
  if opfn is None:
    plt.show()
  else:
    plt.savefig(opfn)

def MutatePlot_PKNAP(arr, opfn=None):
  from matplotlib import font_manager
  myfont = font_manager.FontProperties
  bg = '#1B1A1A'
  plt.rcParams['text.color']= '#000000'
  plt.figure(figsize=(6,3), facecolor=bg)
  ax = plt.subplot(111)
  ax.set_facecolor(bg)
  plt.imshow(arr, cmap='bwr_r', vmin=-2, vmax=2)
  plt.colorbar(label='Predicted '+r'$\Delta\Delta$'+'G (kcal/mol)', ticks=np.arange(-2,3,1), fraction=0.05, shrink=0.83)
  plt.xlabel('Mutated Residue', color='#000000')
  plt.ylabel('Position')
  plt.xticks(np.arange(20), rlist)
  plt.yticks(np.arange(9), np.arange(1,10))
  if opfn is None:
    plt.show()
  else:
    plt.savefig(opfn)

def ddg_ml(dglayer, otindex):
  #return dglayer[otindex[1]] - dglayer[otindex[0]]
  return dglayer[otindex[0]] - dglayer[otindex[1]]
# DEPRECATED
#def ddgse2_ml(stdlayer, otindex):
#  return stdlayer[otindex[1]]**2 + stdlayer[otindex[0]]**2

def Sigmoid(x):
  return 1/(1+np.exp(-x))

#def AUC(x,y):
#  return sm.roc_auc_score(x>0,Sigmoid(y))

