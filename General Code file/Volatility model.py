# Function list
from __future__ import print_function, division
import time
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats as scs
import scipy
from functools import partial
from bekk import (BEKK, ParamStandard, ParamSpatial, simulate_bekk,download_data, plot_data)
from bekk import filter_var_python, likelihood_python
from bekk.recursion import filter_var
from bekk.likelihood import likelihood_gauss
from bekk.utils import take_time
from arch.bootstrap import MCS

# UT
class UT:
    def __init__(self,X,S = 50):

        T,N = X.shape
        self.T = T
        self.N = N
        Sigma = np.asmatrix(np.cov(X.T))
        # cross-validation
        n1 =  int(np.floor(2*T/3))#int(np.floor(T*(1-1/np.log(T))))
        n2 = self.T - n1
        R = np.zeros(N)
        Cmin = np.absolute(np.min(Sigma))*np.sqrt(T/np.log(N))
        Cmax = np.absolute(np.max(Sigma))*np.sqrt(T/np.log(N))
        band = np.linspace(Cmin,Cmax,N)
###SKIP##
        for v in range(S):
            s = np.random.permutation(np.arange(self.T))
            X1 = X[s[np.arange(n1)],:]
            X2 = X[s[np.arange(n1,self.T)],:]
            Sigma1 = np.asmatrix(np.cov(X1.T))
            Sigma2 = np.asmatrix(np.cov(X2.T))
            for i in range(N):
                C = band[i]
                R[i] = R[i]+np.sum(np.linalg.norm(self.Thresholding_func(Sigma1,C) - Sigma2,'fro'))
        C_opt = band[np.where(R == np.min(R))[0][0]]
        w,v = np.linalg.eig(self.Thresholding_func(Sigma,C_opt))
        self.Sigma_hat = self.Thresholding_func(Sigma,C_opt)

    def Thresholding_func(self,Sigma,C):
        tau = C*np.sqrt(np.log(self.N)/self.T)
        sig = Sigma.A1
        ind = np.array([np.absolute(x)<tau for x in sig])
        sig = sig*ind
        return np.asmatrix(np.reshape(sig,(self.N,self.N)))
#UT(X).Sigma_sparse

# AT
class AT:
    def __init__(self,X,thresholding = 'soft',cv = True,N = 10,H = 5):
        n,p = X.shape
        self.thresholding = thresholding
        Sigma = np.cov(X.T)
        R = np.zeros(4*N)
        top = X - np.mean(X,axis = 0)
        top = top.T@top
        middle = -2*np.reshape(np.asmatrix(top).A1*np.asmatrix(Sigma).A1,(p,p))
        bottom = np.square(Sigma)
        top = np.square(top)
        theta = (top+middle+bottom)/n
        if cv == True:
            for v in range(H):
                n1 = np.random.choice(np.arange(10,n-10),1)[0]
                for j in range(4*N):
                    delta = j/N
                    Sigma1 = np.cov(X[np.arange(n1),:].T)
                    Sigma2 = np.cov(X[np.arange(n1,n),:].T)
                    top = X[np.arange(n1),:] - np.mean(X[np.arange(n1),:],axis = 0)
                    top = top.T@top
                    middle = -2*np.reshape(np.asmatrix(top).A1*np.asmatrix(Sigma1).A1,(p,p))
                    bottom = np.square(Sigma)
                    top = np.square(top)
                    theta1 = (top+middle+bottom)/n
                    lamda1 = delta*np.sqrt(theta1*np.log(p)/n)
                    R_hat = self.thresholding_func(Sigma1,lamda1) - Sigma2
                    R[j] = np.sum(R_hat@R_hat.T)**2
            R = R/H
            delta = list(R).index(min(R))/N
        else:
            delta = 2

        lamda = delta*np.sqrt(theta*np.log(p)/n)
        self.Sigma_hat = self.thresholding_func(Sigma,lamda)

    def thresholding_func(self,Sigma,lamda,ita = 1):
        Sigma = np.asmatrix(Sigma)
        dig = Sigma.diagonal()
        lamda = np.asmatrix(lamda)
        if self.thresholding == 'soft':
            rtn = np.reshape(Sigma.A1/np.absolute(Sigma.A1)*(Sigma.A1 - lamda.A1)*np.array([x>0 for x in list(Sigma.A1 - lamda.A1)]),Sigma.shape)
            np.fill_diagonal(rtn,dig)
            return rtn
        if self.thresholding == 'alasso':
            ind = [x>0.0 for x in (1 - np.absolute(Sigma.A1 - lamda.A1)**ita)]
            rtn = np.reshape(Sigma.A1*np.array(ind),Sigma.shape)
            np.fill_diagonal(rtn,dig)
            return rtn
#AT(X,thresholding = 'alasso').Sigma_sparse

# lw
class LW:
    def __init__(self,X):
        T,N = X.shape
        Sigma = np.cov(X.T)
        # demeaned sequence
        mT = np.trace(Sigma)/N
        dT_sqr = np.trace(Sigma@Sigma)/N - np.square(mT)
        X_dot = X- np.mean(X,axis = 0)
        bT_sqr_bar = sum(map(lambda t:np.linalg.norm(X_dot[t,:].T@X_dot[t,:] - Sigma,'fro')**2,range(T)))/(N*T*T)
        bT_sqr = min(bT_sqr_bar,dT_sqr)
        aT_sqr = dT_sqr - bT_sqr
        rho1 = mT*bT_sqr/dT_sqr
        rho2 = aT_sqr/dT_sqr
        self.Sigma_hat = rho1*np.identity(N) + rho2*Sigma

#LW(X).Sigma_sparse

# MT
class MT:
    def __init__(self,X,p = 0.05,procedure = 'Holm',rowwise = False): # Bonferroni
        T,N = X.shape
        Sigma = np.asmatrix(np.cov(X.T))
        D = np.reshape(np.zeros(N*N),(N,N))
        np.fill_diagonal(D,Sigma.diagonal())
        R = np.asmatrix(np.corrcoef(X.T))
        if procedure == 'Bonferroni':
            tau = pow(np.sqrt(T),-1)*scipy.stats.norm.ppf(1 - p/(2*N*(N - 1)/2))
            rho = R.A1
            for s in range(len(R.A1)):
                if np.absolute(R.A1[s])<tau:
                    rho[s] = 0
            R_sparse = np.reshape(rho,R.shape)
        elif procedure == 'Holm':
            rho0 = R.A1
            rho1 = np.zeros(int(N*(N - 1)/2))
            k = 0
            for i in range(N-1):
                for j in range(i+1,N):
                    rho1[k] = R[i,j]
                    k = k + 1
            rho1 = sorted(rho1,reverse = True)
            s = 0
            e = len(rho1) - 1
            m = int(np.floor((e-s)/2))
            interval = (e - s)/2
            while interval>=1:
                if(np.absolute(rho1[m])<scipy.stats.norm.ppf(1 - p/(2*N*(N - 1)/2 - m + 1))):
                    s = m
                    m = int(np.floor((e+s)/2))
                else:
                    e = m
                    m = int(np.floor((e+s)/2))
                interval = (e - s)/2
            if(np.absolute(rho1[e])<scipy.stats.norm.ppf(1 - p/(2*N*(N - 1)/2 - e + 1))):
                m = e
            reject_rho = np.asarray(rho1)[range(m+1)]
            rho0[np.where(rho0 <= max(reject_rho))] = 0
            R_sparse = np.reshape(rho0,R.shape)
        np.fill_diagonal(R_sparse,1)
        self.Sigma_hat = scipy.linalg.sqrtm(D)@R_sparse@scipy.linalg.sqrtm(D)
#MT(X,0.1,'Bonferroni').Sigma_sparse


# BEKK arch_model
def BEKK_method(epsi,restriction = "full"): # "diagonal"
    T,N = epsi.shape
    nstocks = N
    use_target = True
    nobs = T
    restriction = 'full'
    cython = True
    target = np.eye(nstocks)
    A = np.eye(nstocks) * .09**.5
    B = np.eye(nstocks) * .9**.5
    target = np.eye(nstocks)
    param_true = ParamStandard.from_target(amat=A, bmat=B, target=target)
    bekk1 = BEKK(epsi)
    bekk1.estimate(param_start=param_true, restriction=restriction,use_target=use_target, method='SLSQP', cython=cython)
    return bekk1.hvar

# More information: https://github.com/khrapovs/bekk



'''
import importlib.util
spec = importlib.util.spec_from_file_location("Volatility model.py", "D://Thesis20191110//General Code file//BEKK.py")
MG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MG)
MG.BEKK_method
'''
