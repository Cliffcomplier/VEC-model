import numpy as np
import pandas as pd
import scipy.sparse
import scipy.linalg
import scipy
import math
import heapq
import random
import time
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
import matplotlib.pyplot as matplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

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
    return A

class VEC:
    def __init__(self,epsi):
        epsi = epsi.T
        N,T = epsi.shape
        epsi = np.asmatrix(epsi)
        self.epsi = epsi
        self.N,self.T = (N,T)
        self.Ns = int(N*(N+1)/2)
        self.Ns2 = pow(self.Ns,2)
        self.N,self.T = (N,T)
        self.mu,self.eps,self.step = (2,0.05,0.0005)
        self.O = np.asmatrix(np.ones((N,N)))
        self.I = np.identity(N)
        R_list,r_list = ([],[])
        for t in range(T):
            R = np.dot(epsi[:,t],epsi[:,t].T)
            R = self.Enforce_PD(R)
            r_list.append(vech(R))
            R_list.append(R)
        self.r_list = r_list # time list of vech(epsi@epsi.T)
        self.R_list = R_list # time list of epsi@epsi.T
        ## Test
        self.A = np.identity(self.Ns)
        self.B = self.A
        self.C = np.identity(self.N)
    def Enforce_PD(self,A):

        return np.asmatrix(A)
    def cov_mat(self,A,B,C):
        '''
        Dim(A): [N(N+1)/2] * [N(N+1)/2]
        Dim(B): [N(N+1)/2] * [N(N+1)/2]
        Dim(C): [N] * [N]
        A,B,C are np.matrics
        '''
        # Compute the Ht(conditional covariance matrix) with VEC parameters given
        Ns2,Ns,N,T = (self.Ns2,self.Ns,self.N,self.T)
        Ht = np.zeros((N,N,T))
        vecht = np.ones((Ns,T))
        Ht_last = Ht
        Ht[:,:,0] = np.identity(N)
        vecht[:,0] = vech(Ht[:,:,0]).A1
        r_list = self.r_list
        for t in range(1,T):
            d = vech(C) + A@r_list[t-1]+ B@vech(np.asmatrix(Ht[:,:,t - 1])) # r_list saves vech(epsi@epsi.T)
            Ht_now = restack(d)
            #Ht_now = EVD_Threshold(Ht_now,0)
            Ht_now = self.Enforce_PD(Ht_now)
            Ht[:,:,t] = Ht_now
            # Test only: print(min(np.linalg.eig(Ht[:,:,t])[0])>0)
            vecht[:,t] = vech(Ht_now).A1
        return Ht
    def derivative(self,Ht): # With majorization
        Ns2,Ns,N,T = (self.Ns2,self.Ns,self.N,self.T)
        epsi = self.epsi
        da = np.zeros((Ns,Ns))
        db = np.zeros((Ns,Ns))
        dc = np.zeros((N,N))
        R_list = self.R_list
        r_list = self.r_list
        h_last = vech(Ht[:,:,0])
        pickle.dump(r_list,open("D://debug//r_list.txt","wb"))
        for t in range(1,T):
            if np.linalg.det(Ht[:,:,t])<=0:
                da = np.zeros((Ns,Ns))
                db = np.zeros((Ns,Ns))
                dc = np.zeros((Ns,1))
                return [da,db,restack(dc)]
            invH = np.linalg.inv(Ht[:,:,t])
            invH = np.asmatrix(invH)
            dH = invH.T + R_list[t]
            dH = dH/(1e200*N*N)
            pickle.dump(dH,open("D://debug//dH%d.txt"%t,"wb"))
            dc = dc + dH
            dh = vech(dH)
            pickle.dump(h_last,open("D://debug//h%d_list.txt"%t,"wb"))
            da = da + np.kron(dh,r_list[t-1].T)
            h_last = vech(Ht[:,:,t-1])
            db = db + np.kron(dh,h_last.T)
        return [da,db,dc]
    def Loss_func(self,Ht):
        # Computation the approximation of logliklihood function of Ht
        # It is one part of loss function (what the program wants to minimize)
        T = self.T
        mu = self.mu
        R_list = self.R_list
        for t in range(T):
            if np.linalg.det(Ht[t,:,:])<=0:
                return np.nan
        L = list(map(lambda t:np.log(np.linalg.det(Ht[t,:,:]))+np.trace(np.linalg.inv(Ht[t,:,:]@R_list[t])),range(T)))
        L = np.sum(L)/T
        return L.real
    def estimation(self,lam,maxtie = 100):
        Ns2,Ns,N,T = (self.Ns2,self.Ns,self.N,self.T)
        mu,eps,step = (self.mu,self.eps,self.step)
        # 0. Initial value
        ZD = list(map(lambda dim: 1e-3*np.asmatrix(np.identity(dim)),[Ns,Ns,N]))
        ZL = list(map(lambda dim: 1e-3*Low_rank(dim,0.5),[Ns,Ns,N]))#return value of Low_rank is matrix
        Z = list(map(lambda D,L: D+L,ZD,ZL))
        #list(map(lambda z:print(z.shape),Z))
        Z= list(map(lambda mat:EVD_Threshold(mat,0),Z))
        #pd = list(map(lambda z:min(np.linalg.eig(z)[0])>0,Z))
        #list(map(lambda z:print(z.shape),Z))
        Ht = self.cov_mat(Z[0],Z[1],Z[2])
        Ht0 = Ht
        Lambda = list(map(lambda N: np.asmatrix(np.ones((N,N))),[Ns,Ns,N]))
        for it0 in range(maxtie):
            # 1. Zeta step (level 3)
            print("it0 = %d"%it0)
            Zeta = list(map(lambda Z_mat,Lambda_mat:Z_mat+mu*Lambda_mat,Z,Lambda))
            Zeta = list(map(lambda mat:EVD_Threshold(mat,0),Zeta))
            # 2. Theta step
            ## ZL: Low rank step
            for it1 in range(maxtie):
                Ht0 = Ht
                for it2 in range(maxtie):
                    df = self.derivative(Ht)
                    magn = list(map(lambda d: np.sum(np.absolute(d)),df))
                    # stop measures nonzeros in differention
                    if sum(magn) != 0:
                        df = list(map(lambda d,base:step*d/base,df,magn))
                        # Gradient descent
                        ZL = list(map(lambda L,d:L - d,ZL,df))
                        # SVD: low rank
                        ZL = list(map(lambda L:SVD_Threshold(L,step*lam),ZL))

                        # Gradient descent
                        ZD = list(map(lambda D,d:D - np.diag(np.diag(d)),ZD,df))
                        # EVD: sparse
                        ZD = list(map(lambda D:EVD_Threshold(D - step*lam*np.identity(D.shape[0]),0.),ZD))
                        # The last line is the same as SVD:low rank, but because of the difference of python function, they look different
                        Z = list(map(lambda D,L: D+L,ZD,ZL))
                        Ht = self.cov_mat(Z[0],Z[1],Z[2])
                    else:
                        break
            # 3. Lambda step (level 3)
            for i in range(3):
                Lambda[i] = Lambda[i] - (1/mu)*(Zeta[i] - Z[i])
        return Ht.T
    def tunning(self):
        Min,Max = (0,5)
        lam1 = np.linspace(Min,Max,20)
        lam1 = pow(10,lam1)
        res1 = Parallel(n_jobs=20)(delayed(self.estimation)(i) for i in lam1)
        signature = 0
        for dgpno in [1,2,3,4]:
            if not os.path.exists("res1_(%d,%d)_%d.txt"%(self.N,self.T,dgpno)):
                signature = dgpno
                break

        pickle.dump(res1,open("res1_(%d,%d)_%d.txt"%(self.N,self.T,signature),"wb"))
        min_res = 1e10
        min_no = 0
        no = 0
        for res in res1:
            loss = self.Loss_func(res)
            if not (np.isnan(loss) or np.isnan(loss)):
                if min_res>loss:
                    min_res = loss
                    min_no = no
            no = no + 1
        pickle.dump(lam1,open("lam1_(%d,%d)_%d.txt"%(self.N,self.T,signature),"wb"))
        return res1[min_no]
'''os.chdir("D://debug") # File location

N,T = (10,100)
for No in [1,2,3,4]:
    Obj = DGPs(N,T,No)
    Ht = Obj.Ht
    pickle.dump(Ht,open("20191110 Ht_(%d,%d,%d)"%(No,N,T),"wb"))
    epsi = Obj.epsi
    Ht_hat = VEC(epsi).tunning()
    pickle.dump(Ht_hat,open("Sim20191110plify Ht_hat_(%d,%d,%d)"%(No,N,T),"wb"))
    pickle.dump(Norm(Ht,Ht_hat),open("D://New DGP Norm//Norm with DGP%d(%d,%d) and method VEC(precision).txt"%(No,N,T)))
'''
