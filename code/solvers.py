#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy.linalg as la
# import time
# import matplotlib.pyplot as plt
import scipy.optimize as sciop
import numpy as np
from scipy import linalg
# from mne.datasets import sample
# from mne.viz import plot_sparse_source_estimates
# from sklearn import linear_model
# from celer import MultiTaskLasso

def fista(opt):
    lam = opt['lam']
    maxits = opt['maxits']
    X = opt['X']
    b = opt['b']   
    flag = opt['flag']            
    z = X.T @ b
    #A = (X.T @ X);
    L = opt['L']
    x = opt['x0']
    p = X.shape[0]
 
    def prox(x,T):
        no = np.sqrt(np.sum(x**2,1));
        x = np.vstack(np.maximum(no-T,0)/no)*x;
        x[np.isnan(x)]=0
        return x
    
    def objfun(x):
        v =np.sum(np.sqrt( np.sum(x**2,1)))
        f = lam*v + 0.5*la.norm(np.vstack(X@x) - np.vstack(b),'fro')**2
        return f/p
    

    tol = opt['tol']
    
    y = x
    theta =1
    gamma = 1/L
    res = np.zeros([maxits,1])
    fvals = np.zeros([maxits,1])
    for it in range(maxits):
        xm = x
        if flag==0:
            grad = X.T@(X@x) - z
            x = prox(x - gamma*grad,  gamma*lam)
        else:
            grad = X.T@(X@y) - z
            x = prox(y - gamma*grad,  gamma*lam)
            theta = (1+np.sqrt( 1+ 4* theta**2 ))/2  
        
            ym = y;
            aa = (theta-1)/theta 
            y = x + aa*(x - xm)
            
            if flag==2 and np.sum( (ym-x)*(x-xm) ) > 0:
                y = x
                theta = 1

        r = la.norm(x-xm,2);
        res[it] = r
        if r< tol:
            break
         
        fvals[it] = objfun(x)
    res = res[:it]
    fvals = fvals[:it]
    return x,fvals




def LassoPro(X,Y, lam,maxits=1000,maxcor= 10):
    Xt = X.T
    Y=-Y
    n = Y.shape[0]
    m = X.shape[1]
    p = X.shape[0]
    fvals = [];
    y=np.vstack(Y)
    def objfn(t):
        M = lam*np.eye(n)+np.dot(X, np.vstack(t**2)* Xt)
        alpha = lam*linalg.solve(M, Y)
        v = t* np.sum(np.dot(Xt,alpha)**2,1);
        al=np.vstack(alpha)
        fval = -0.5*linalg.norm(al)**2 - 0.5/lam*np.sum(t*v)+lam/2*linalg.norm(t)**2 + np.sum(al*y);
        grad = lam*t - 1/lam*v
        fvals.append(fval/p)
        return fval, grad
    

    
    #run lbfgs
    # myopts ={'ftol': 1e-03, 'gtol': 1e-03, 'eps': 1e-03}
    # myopts ={'iprint': 1, 'gtol': 1e-03}
    myopts = { 'gtol': 1e-03,'maxiter':maxits,'maxcor': maxcor}
    t0 = np.random.randn(m,1);
    
    t = sciop.minimize(objfn, t0, method='L-BFGS-B', jac=True,options=myopts)

    x = t.x
    t2 = np.vstack(x**2)
    W = np.eye(n)+1/lam *np.dot(X, t2* Xt)
    alpha = linalg.solve(W, Y)
    X= -1/lam* t2*np.dot(Xt,alpha)
    
 
    return X, fvals





def LassoPro_Cov(Q,z, lam,maxits=1000,maxcor= 10):
    fvals = [];
    n = len(z)
    Q = np.squeeze(np.asarray(Q))
    def objfn(t):
        tz =t* z
        # tt = np.vstack(t)@np.vstack(t).T
        # a1 = linalg.solve( lam*np.eye(n)+Q*tt, tz)
        a1 = linalg.solve( lam*np.eye(n)+Q*t[:,None]*t[None,:], tz)
        # a1 = linalg.solve( lam*np.eye(n)+Q * t[:,None] * t[None,:], tz)
        # a1 = linalg.solve( lam*np.eye(n)+np.diag(t)@Q@np.diag(t), tz)
        x = t*a1
        Qx = np.ravel(Q@x);
        # Qx = Q@x
        grad = lam*t + a1*(Qx - z)
        f = lam*linalg.norm(a1)**2/2. + lam*linalg.norm(t)**2/2. +np.sum(x*Qx)/2. - np.sum(x*z)
        fvals.append(f)
        return f, grad
    

    
    #run lbfgs
    # myopts ={'ftol': 1e-03, 'gtol': 1e-03, 'eps': 1e-03}
    # myopts ={'iprint': 1, 'gtol': 1e-03}
    myopts = { 'gtol': 1e-03,'maxiter':maxits,'maxcor': maxcor}
    t0 = np.random.randn(n,1);
    
    t = sciop.minimize(objfn, t0, method='L-BFGS-B', jac=True,options=myopts)
    t0 = t.x;
    t = np.vstack(t0)
    # tt = t@t.T
    # a = linalg.solve( lam*np.eye(n)+Q*tt, t0*z)
    a = linalg.solve( lam*np.eye(n)+Q*t0[:,None]*t0[None,:], t0*z)
    x = t0*a
 
    return x, fvals




def LassoPro2(X,Y, lam,maxits=1000,maxcor= 10):
    Xt = X.T
    Y=-Y
    n = Y.shape[0]
    m = X.shape[1]
    fvals = [];
    y=np.vstack(Y)
    def objfn(t):
        t2 = t**2
        M = lam*np.eye(n)+np.dot(X, t2[:,None]*  Xt)
        alpha = lam*linalg.solve(M, Y)
        v = t* (Xt@alpha)**2

        fval = -0.5*linalg.norm(alpha)**2 - 0.5/lam*np.sum(t*v)+lam/2*linalg.norm(t)**2 + np.sum(alpha*y);
        grad = lam*t - 1/lam*v
        fvals.append(fval/n)
        return fval, grad
    

    
    #run lbfgs
    # myopts ={'ftol': 1e-03, 'gtol': 1e-03, 'eps': 1e-03}
    # myopts ={'iprint': 1, 'gtol': 1e-03}
    myopts = { 'gtol': 1e-03,'maxiter':maxits,'maxcor': maxcor}
    t0 = np.random.randn(m,1);
    
    t = sciop.minimize(objfn, t0, method='L-BFGS-B', jac=True,options=myopts)

    x = t.x
    t2 = x**2 
    M = lam*np.eye(n)+np.dot(X, t2[:,None]* Xt)
    alpha = linalg.solve(M, Y)
    X= -1/lam* t2*(Xt@alpha)
    
 
    return X, fvals


