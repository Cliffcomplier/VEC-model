import numpy as np
import pandas as pd
import scipy.sparse
import scipy.linalg
import scipy
import math

import time
import matplotlib
import random
from arch.univariate import arch_model
from scipy import optimize
import sklearn.preprocessing
from sklearn import linear_model as lm
from joblib import Parallel, delayed
from functools import reduce
from itertools import combinations
import warnings
import pickle
import os, sys
def vech(X):
    # Vectorizing matrix
    N,N = X.shape
    vech = np.ones(int(N*(N+1)/2))
    k = 0
    for i in range(0,N):
        vech[k:k + N - i] = X[i,i:N]
        k = k + N - i
    vech = np.asmatrix(vech).T
    return vech
def restack(vecx):
    # Revert the vch to a symmetric matrix
    Ns = len(vecx)
    N = int((np.sqrt(1+8*Ns) - 1)/2)
    X = np.asmatrix(np.reshape(np.ones(N*N),(N,N)))
    X[np.triu_indices(N)] = np.reshape(vecx,int((N*(N+1)/2),))
    X[np.tril_indices(N)] = X.T[np.tril_indices(N)]
    return X
def EVD_Threshold(A,epsilon):
    # Large_GARCH eq(15) and eq(16)
    u,vh = np.linalg.eig(A)
    N,N = A.shape
    B = np.zeros((N,N))
    for i in range(N):
        v = np.asmatrix(vh[:,i]) # Noted v.shape = (N,1)
        B = max(u[i],epsilon)*v@v.T
    return B
def SVD_Threshold(A,epsilon):
    # Large_GARCH eq(23)
    N,N = A.shape
    A = np.asarray(A)
    u,s,vh = np.linalg.svd(A)
    sm = s - epsilon
    sm[sm<=0] = 0.
    return np.dot(u * sm, vh)
def Low_rank(N,ratio):
    import heapq
    A = np.random.normal(0,1,(N,N))
    u,s,vh = np.linalg.svd(A)
    smax = heapq.nlargest(int(N*ratio),s)[-1]
    sm = s
    sm[sm<=smax] = 0.
    return np.asmatrix(np.dot(u * sm, vh))
def Enforce_PD(A):
    eig,eigh = np.linalg.eig(A)
    min_eig = np.min(eig)
    while min_eig<0 or min_eig ==0:
        eig[eig<=0] = 0.01
        A = eigh@np.diag(eig)@eigh.T
        eig,eigh = np.linalg.eig(A)
        min_eig = np.min(eig)
        A = np.asmatrix(A)
    return A
print("DGPs(N,T,dgp_no) with objs{epsi,Ht,A,B,C}")
class DGPs:
    # H matrix's shape (N,N,T)
    # epsi's shape (N,T)
    def par1(self,d):
        betamin,betamax = (self.betamin,self.betamax)
        Omega_diag = np.random.uniform(0,1,3)
        Omega = np.diag(Omega_diag)
        for i in range(3):
            for j in range(i+1,3):
                Omega[i,j] = np.absolute(Omega[i,i] - Omega[j,j])
                Omega[i,j] = pow(0.2,Omega[i,j])
        Omega = Omega + Omega.T - np.diag(Omega_diag)
        beta = [list(np.random.uniform(betamin,betamax,d)),list(np.random.uniform(betamin,betamax,d)),np.random.normal(0.,0.25,d)]
        beta = np.asmatrix(beta).T
        Sigma_diag = np.random.normal(0,0.25,d)#np.random.chisquare(3,d)
        Sigma_diag = np.absolute(Sigma_diag)
        Z0 = beta@Omega@beta.T + np.diag(Sigma_diag)
        return Z0
    def par2(self,d):
        betamin,betamax = (self.betamin,self.betamax)
        rz = random.choice([1,2,3,4])
        Omega_diag = np.random.uniform(0,1,rz)
        Omega = np.diag(Omega_diag)
        for i in range(rz):
            for j in range(i+1,rz):
                Omega[i,j] = np.absolute(Omega[i,i] - Omega[j,j])
                Omega[i,j] = pow(0.2,Omega[i,j])
        Omega = Omega + Omega.T - np.diag(Omega_diag)
        beta = list(map(lambda i:np.random.uniform(betamin,betamax,d),range(rz)))
        beta = np.asmatrix(beta).T
        Sigma_diag = np.random.normal(0,0.25,d)#np.random.chisquare(3,d)
        Sigma_diag = np.absolute(Sigma_diag)
        Z0 = beta@Omega@beta.T + np.diag(Sigma_diag)
        return Z0
    def par3(self,d):
        # Uniform
        betamin,betamax = (self.betamin,self.betamax)
        r1 = random.choice([1,2,3,4])
        Omega_diag = np.random.uniform(0,1,r1)
        Omega = np.diag(Omega_diag)
        for i in range(r1):
            for j in range(i+1,r1):
                Omega[i,j] = np.absolute(Omega[i,i] - Omega[j,j])
                Omega[i,j] = pow(0.2,Omega[i,j])
        Omega = Omega + Omega.T - np.diag(Omega_diag)
        beta = list(map(lambda i:np.random.uniform(betamin,betamax,d),range(r1)))
        beta = np.asmatrix(beta).T
        Z0 = beta@Omega@beta.T
        # Normal
        r2 = random.choice([1,2,3,4])
        Omega_diag = np.random.uniform(0,1,r2)
        Omega = np.diag(Omega_diag)
        for i in range(r2):
            for j in range(i+1,r2):
                Omega[i,j] = np.absolute(Omega[i,i] - Omega[j,j])
                Omega[i,j] = pow(0.2,Omega[i,j])
        Omega = Omega + Omega.T - np.diag(Omega_diag)
        beta = list(map(lambda i:np.random.normal(betamin,betamax,d),range(r2)))
        beta = np.asmatrix(beta).T
        Sigma_diag = np.random.normal(0,0.25,d)#np.random.chisquare(3,d)
        Sigma_diag = np.absolute(Sigma_diag)
        Z0 = Z0 + beta@Omega@beta.T + np.diag(Sigma_diag)
        return Z0/2
    def par4(self,d):
        omg = np.random.uniform(0,1,1)[0]
        sig = np.random.chisquare(3,1)[0]
        Omega = np.asmatrix([[omg,0,omg],[omg,0,0],[0,sig,0]])
        beta = [list(np.random.uniform(0.25,2,d)),list(np.random.uniform(0.25,2,d)),np.random.normal(0.,0.25,d)]
        beta = np.asmatrix(beta).T
        Sigma_diag = np.random.normal(0,0.25,d) #np.random.chisquare(3,d)
        Sigma_diag = np.absolute(Sigma_diag)
        Z0 = beta@Omega@beta.T + np.diag(Sigma_diag)
        return Z0
    def __init__(self,N,T,dgp):
        self.N,self.T = (N,T)
        self.betamin,self.betamax = (0.25,1)
        self.Ns = int(N*(N+1)/2)
        N,T,Ns = (self.N,self.T,self.Ns)
        if dgp == 1:
            A,B,C = list(map(lambda d:self.par1(d),[Ns,Ns,N]))
        elif dgp == 2:
            A,B,C = list(map(lambda d:self.par2(d),[Ns,Ns,N]))
        elif dgp == 3:
            A,B,C = list(map(lambda d:self.par3(d),[Ns,Ns,N]))
        elif dgp == 4:
            A,B,C = list(map(lambda d:self.par4(d),[Ns,Ns,N]))
    # Generate the random parameter
        C = Enforce_PD(C)
        C = 1e-2*vech(C)
        A = 1e-3*Enforce_PD(A)
        B = 1e-3*Enforce_PD(B)
        print(np.min(np.linalg.eig(A)[0]))
        print(np.min(np.linalg.eig(B)[0]))
        print(np.min(np.linalg.eig(restack(C))[0]))
    # Setting the initial value for Ht[:,:,0]
        Ht = []
        ht = []
        epsi = np.zeros((N,T))
        Ht.append(np.asmatrix(np.identity(N)))
        ht.append(vech(Ht[0]))
        epsitn = np.random.normal(0,1,(N,1))
        epsitn = np.asmatrix(epsitn)
        epsitn.shape = (N,1)
        epsi[:,0] = epsitn.A1
        for t in range(1,T):
            epsitm = epsi[:,t-1]
            epsitm = np.reshape(epsitm,(N,1)) # tm: t minus 1
            htm = ht[t-1]
            second = np.dot(epsitm,epsitm.T) #htn: ht now
            E = restack(C + np.dot(A,vech(second)) + np.dot(B,ht[t-1]))
            E = Enforce_PD(E)
            Ht.append(E)
            ht.append(vech(E))
            epsitn = scipy.linalg.sqrtm(Ht[t])@np.random.normal(0,1,(N,1))
            epsi[:,t] = np.reshape(epsitn,(N,))
        self.Ht = np.zeros((N,N,T))
        for t in range(T):
            self.Ht[:,:,t] = Ht[t]
        self.epsi = epsi.T
        self.Ht = self.Ht.T
        self.A = A
        self.B = B
        self.C = C
'''
import importlib.util
spec = importlib.util.spec_from_file_location("DGPs ", "D://Thesis20191110//General Code file//DGPs.py")
DGPs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DGPs )
'''
