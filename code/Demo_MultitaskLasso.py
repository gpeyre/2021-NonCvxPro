#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import io
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from solvers import LassoPro
from celer import MultiTaskLasso
import time
from scipy.io import savemat,loadmat


mypath = 'GroupLassoDatasets/'  # where to save datasets
mycelerpath = 'CelerGroupLassoResults/'  # where to save results


fac_ = 50 #For paper, choose 10,20,50,100

dataset_id=1


if dataset_id == 0:    
    rng = np.random.RandomState(42)
    dataset = "synthetic-wave"
    n_samples, n_features, n_tasks = 300, 1000, 100
    
    n_relevant_features = 5
    coef = np.zeros((n_tasks, n_features))
    times = np.linspace(0, 2 * np.pi, n_tasks)
    
    indx = np.random.permutation(n_features)
    for k in range(n_relevant_features):
        coef[:, indx[k]] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))
    
    X = rng.randn(n_samples, n_features)
    Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
if dataset_id == 1:
    dataset = "synthetic"
       
    rng = np.random.RandomState(40)
    n_samples, n_features, n_tasks = 50, 1200, 20
    # n_samples, n_features, n_tasks = 30, 1000, 100
    
    n_relevant_features = 10
    coef = np.zeros((n_tasks, n_features))
    times = np.linspace(0, 2 * np.pi, n_tasks)
    
    indx = np.random.permutation(n_features)
    for k in range(n_relevant_features):
        coef[:, indx[k]] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))
    
    X = rng.randn(n_samples, n_features)
    Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)

    
    


if dataset_id == 2:
    # The data can be found in https://drive.google.com/open?id=139nKKy0AkpkZntB80n-LmcuzGgC8pQHi
    # Please unzip the file "meg_data.tar.gz"
    dataset = 'meg_full'
    data = loadmat('GroupLassoDatasets/meg_Xy_full.mat')
    X = np.array(data['X'], dtype=np.float, order='F')
    Y = np.array(data['Y'], dtype=np.float)
    y = Y

    
if dataset_id==3:
    #Install MNE and run the mne_data.py script to download the data from MNE website.
    data = loadmat('GroupLassoDatasets/MNE_Data.mat')
    dataset = 'MNE'
    X = data['X']
    Y = data['Y']
    y = Y
    
# Save dataset if synthetic
if dataset_id <2:
    mdic = { "X": X, "Y": Y}
    savemat(mypath+dataset+".mat", mdic)
    


lammax = np.sqrt(np.max(np.sum((X.T@Y)**2,1)   ))
lam = lammax/fac_


maxfun = lambda z: np.sqrt(np.sum(np.abs(z)**2,1))
cert = lambda w: maxfun((X.T@(X@w) - X.T@Y)/lam);

rngx0 = np.random.RandomState(2)
# x0 = rngx0.randn(X.shape[1],Y.shape[1])


# Nonconvex Pro
t = time.time()
a0,fvals0 = LassoPro(X,Y, lam,maxits=50)   
elapsed = time.time() - t
print('mtlPro...')
print('time:', elapsed)
print('obj val:', fvals0[-1:])
print('optimality',np.max(cert(a0)))

# %%
# run celer and record io output
old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout
t = time.time()
a = MultiTaskLasso(alpha = lam/X.shape[0],verbose=True,fit_intercept=False).fit(X,Y).coef_.T
elapsed2 = time.time() - t

# retrieve objective values
output = new_stdout.getvalue()
output = output.split('primal ')
celer_obj = np.zeros((len(output)-1,1))
for i in range(len(output)-1):
    x = output[i+1].split(', ')
    celer_obj[i] = float(x[0])

sys.stdout = old_stdout

print('celer...')
print('time:', elapsed2)
print('obj val:', celer_obj[-1])
print('optimality:',np.max(cert(a)))


# save to .mat file
# objective saved is lambda*|a|_{12} + 1/2 |X*a - Y|_fro^2
mdic = {"objectives": celer_obj*X.shape[0], "times": elapsed2,  "lam": lam, "x": a}
savemat(mycelerpath+dataset+str(fac_)+".mat", mdic)


obj_best = np.minimum(np.min(fvals0),np.min(celer_obj))
plt.semilogy(np.linspace(0, elapsed, num=len(fvals0)),fvals0-obj_best,color='blue', linestyle='dashed',  linewidth=2)
plt.semilogy(np.linspace(0, elapsed2, num=len(celer_obj)),celer_obj-obj_best,'r',  linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Objective value error', fontsize=14)
plt.legend(['Noncvx-Pro','CELER','FISTA'],fontsize=14)
plt.savefig(mypath+dataset+str(fac_)+'.png')

plt.show()

