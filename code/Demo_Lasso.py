
from numpy.linalg import norm
from numpy.matlib import repmat
from libsvmdata import fetch_libsvm
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from solvers import LassoPro,LassoPro_Cov
from celer import Lasso as CELER_LASSO
from sklearn.linear_model import LassoCV
import time
import sys
from io import StringIO

mydatapath = 'LassoDatasets/' #folder to save dataset
mycelerpath = 'CelerLassoResults/' #folder to save celer results

fac_ = -1  #  for cross validation regularisation
# fac_ = 2 # factor to divide largest parameter by
# fac_ = 10 
# fac_ = 20
# fac_ = 50
# fac_ = 100
# dataset = 'housing'
dataset = 'mnist'
dataset = 'a8a'
# dataset = 'w8a'
# dataset = 'leukemia'
# dataset = 'abalone'
# dataset = 'cadata'
# dataset = 'connect-4'


X,y = fetch_libsvm(dataset)
m = X.shape[0]
# remove mean and normalise
y -= y.mean()
Z = repmat(np.mean(X,axis=0), m, 1)
X = X-Z
Z = repmat(norm(X,axis=0), m, 1)
X = np.nan_to_num(X/Z)


if fac_>0:
    lam =np.max( np.abs(X.T@y))/fac_
else:
    # using sklearn to cross validate
    reg = LassoCV(cv=10, random_state=0,fit_intercept=False).fit(X, y)
    alpha_ = reg.alpha_
    lam = alpha_ * m
    
    
    
mdic = {"X": X, "Y": y}
io.savemat(mydatapath+dataset+".mat", mdic)


# %%
Q = X.T@X
z = np.ravel(X.T@y)
print('ncvx-pro...')
if X.shape[0]<X.shape[1]:
    X1 = np.squeeze(np.asarray(X))
    t = time.time()
    apro,fpro = LassoPro(X1,y[:,None], lam,maxits=1000,maxcor= 5)
    elapsedpro= time.time() - t
    fvals= norm(X@apro - y[:,None])**2/(m*2.) + lam/m*norm(apro,1)
    Qa = np.ravel(Q@apro)  
else:        
    t = time.time()
    apro,fpro = LassoPro_Cov(Q,z, lam,maxits=1000,maxcor= 5)
    elapsedpro= time.time() - t
    fvals= norm(X@apro - y)**2/(m*2.) + lam/m*norm(apro,1)
    Qa = np.ravel(Q@apro)
    fpro = (fpro + norm(y)**2/2)/m

print('time:', elapsedpro)
print('obj val:', fvals)
print('optimality:',np.max(np.abs(Qa - z))/lam)
#%%


# run CELER and record io output
old_stdout = sys.stdout
new_stdout = StringIO()
sys.stdout = new_stdout
t = time.time()
a = CELER_LASSO(alpha = lam/m, verbose=True, fit_intercept=False).fit(X,y).coef_.T
elapsed = time.time() - t
Qa = np.ravel(Q@a)

# retrieve the objective values
output = new_stdout.getvalue()
output = output.split('primal ')
celer_obj = np.zeros((len(output)-1,1))
for i in range(len(output)-1):
    x = output[i+1].split(', ')
    celer_obj[i] = float(x[0])
sys.stdout = old_stdout

print('celer...')
print('time:', elapsed)
print('obj val:', celer_obj[-1])
print('optimality:',np.max(np.abs(Qa-z))/lam)

# save to .mat file
mdic = {"objectives": celer_obj*X.shape[0], "times": elapsed,  "lam": lam, "x": a}
io.savemat(mycelerpath+dataset+str(fac_)+".mat", mdic)


# plot
obj_best = np.minimum(np.min(fpro),np.min(celer_obj))
plt.semilogy(np.linspace(0, elapsedpro, num=len(fpro)),fpro-obj_best,color='blue', linestyle='dashed',  linewidth=2)
plt.semilogy(np.linspace(0, elapsed, num=len(celer_obj)),celer_obj-obj_best,'r',  linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Objective value error', fontsize=14)
plt.legend(['Noncvx-Pro','CELER'],fontsize=14)
plt.savefig(mycelerpath+dataset+str(fac_)+'.png')
plt.show()


    
