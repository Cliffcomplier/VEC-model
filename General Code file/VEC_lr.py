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
from datetime import date
# User self define
txt = "D:Research//"
Father_loc = "%s//Thesis%s//"%(txt,date.today())

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
def Max_func(X):
    X[X<=0] = 0
    return X
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

class VEC:
    def __init__(self,epsi):
        if not os.path.exists(Father_loc+"Program Analysis//"):
            os.makedirs(Father_loc+"Program Analysis//")
        if not os.path.exists(Father_loc+"Parameter//"):
            os.makedirs(Father_loc+"Parameter//")
        epsi = epsi.T
        N,T = epsi.shape
        epsi = np.asmatrix(epsi)
        self.epsi = epsi
        self.N,self.T = (N,T)
        self.Ns = int(N*(N+1)/2)
        self.Ns2 = pow(self.Ns,2)
        self.N,self.T = (N,T)
        self.mu,self.eps,self.step = (2,0.001,0.001) #eps = 0.02 step = 0.01
        self.O = np.asmatrix(np.ones((N,N)))
        self.I = np.identity(N)
        R_list,r_list = ([],[])
        for t in range(T):
            R = np.dot(epsi[:,t],epsi[:,t].T)
            #R = self.Enforce_PD(R)
            r_list.append(vech(R))
            R_list.append(R)
        self.r_list = r_list # time list of vech(epsi@epsi.T)
        self.R_list = R_list # time list of epsi@epsi.T
        ## Test
        self.A = np.identity(self.Ns)
        self.B = self.A
        self.C = np.identity(self.N)
    def Enforce_PD(self,A):
        eig = np.linalg.eig(A)[0]
        eigh = np.linalg.eig(A)[1]
        min_eig = np.min(eig)
        nt = 0
        pn = 0.01 #possitive number
        while min_eig<0 or min_eig ==0:
            print(nt)
            eig[eig<=0] = 0.1
            nt = nt + 1
            A = eigh@np.diag(eig)@np.linalg.inv(eigh)
            eig = np.linalg.eig(A)[0]
            eigh = np.linalg.eig(A)[1]
            min_eig = np.min(eig)
        A = np.asmatrix(A)
        return A
    def cov_mat(self,A,B,C):
        '''
        Dim(A): [N(N+1)/2] * [N(N+1)/2]
        Dim(B): [N(N+1)/2] * [N(N+1)/2]
        Dim(C): [N] * [N]
        A,B,C are np.matrics
        '''
        A = self.Enforce_PD(A)
        B = self.Enforce_PD(B)
        C = self.Enforce_PD(C)
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
            Ht[:,:,t] = Ht_now
            # Test only: print(min(np.linalg.eig(Ht[:,:,t])[0])>0)
            vecht[:,t] = vech(Ht_now).A1
        return Ht
    def derivative(self,Ht0,Ht): # With majorization
        Ns2,Ns,N,T = (self.Ns2,self.Ns,self.N,self.T)
        epsi = self.epsi
        da = np.zeros((Ns,Ns))
        db = np.zeros((Ns,Ns))
        dc = np.zeros((Ns,1))
        O = self.O
        I = self.I
        Error = False
        R_list = self.R_list
        for t in range(1,T):
            invH0 = np.linalg.inv(Ht0[:,:,t])
            invH = np.linalg.inv(Ht[:,:,t])
            R = R_list[t]#np.dot(epsi[:,t],epsi[:,t].T)
            h_last = np.asmatrix(vech(np.asmatrix(Ht[:,:,t-1])))
            R_last = R_list[t-1]
            p1 = np.multiply(vech(invH0),vech(2*O - I))
            p2 = np.multiply(vech(invH@R@invH),vech(2*O - I))
            da = da + np.kron(p1-p2,h_last.T)
            db = db + np.kron(p1 - p2,vech(R_last).T)
            dc = dc + p1 - p2
            # This if stream exists since there might be something generating NaNs
        Error = np.sum(np.isnan(da)) or (np.sum(np.isnan(db)) or np.sum(np.isnan(dc)))
        if Error == True:
            #print("Error occures in derivatives function.")
            da = np.zeros((Ns,Ns))
            db = np.zeros((Ns,Ns))
            dc = np.zeros((Ns,1))
        return [da,db,restack(dc)]
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
        GD_t,Maj_t,Theta_t,Zeta_t,Lambda_t,Ini_t,Derivative_t,cov_mat_t = list(map(lambda i : [],range(8))) #*********analysis*********
        ADMM_it,Maj_it,GD_it =  list(map(lambda i :[],range(3))) #*********analysis*********
        Total_t = time.time() #*********analysis*********
        Ns2,Ns,N,T = (self.Ns2,self.Ns,self.N,self.T)
        mu,eps,step = (self.mu,self.eps,self.step)
        # 0. Initial value
        for dgpno in [1,2,3,4]:
            Z_name = Father_loc + "Parameter//Z_lam = %lf_VEC%d(%d,%d).txt"%(lam,dgpno,T,N) # User self define
            if not os.path.exists(Z_name):
                break
        ZD = list(map(lambda dim: 1e-3*np.asmatrix(np.identity(dim)),[Ns,Ns,N]))
        ZL = ZD
        #ZL = list(map(lambda dim: 1e-3*Low_rank(dim,0.5),[Ns,Ns,N]))#return value of Low_rank is matrix
        Z = list(map(lambda D,L: D+L,ZD,ZL))
        #list(map(lambda z:print(z.shape),Z))
        Z = list(map(lambda mat:EVD_Threshold(mat,0),Z))
        #pd = list(map(lambda z:min(np.linalg.eig(z)[0])>0,Z))
        #list(map(lambda z:print(z.shape),Z))
        Ht = self.cov_mat(Z[0],Z[1],Z[2])
        Ht0 = Ht
        Lambda = list(map(lambda N: np.asmatrix(np.ones((N,N))),[Ns,Ns,N]))
        for it0 in range(maxtie):

            lastADMM = Z
            # 1. Zeta step (level 3)
            Zeta_t_temp = time.time() #*********analysis*********
            print("it0 = %d"%it0) #*********analysis*********
            Zeta = list(map(lambda Z_mat,Lambda_mat:Z_mat+mu*Lambda_mat,Z,Lambda))
            #Zeta = list(map(lambda mat:EVD_Threshold(mat,0),Zeta))
            Zeta = list(map(lambda mat:self.Enforce_PD(mat),Zeta))
            Zeta_t_temp = time.time() - Zeta_t_temp #*********analysis*********
            Zeta_t.append(Zeta_t_temp)#*********analysis*********
            # 2. Theta step
            Theta_t_temp = time.time() #*********analysis*********
            ## ZL: Low rank step
            ADMM_it_temp = it0 #*********analysis*********
            Maj_t_temp= time.time()#*********analysis*********
            for it1 in range(maxtie):
                lastMaj = Z
                Ht0 = Ht
                Maj_it_temp = it1 #*********analysis*********
                GD_t_temp = time.time()  #*********analysis*********
                for it2 in range(maxtie):
                    lastGD = Z
                    GD_it_temp = it2 #*********analysis*********
                    df = self.derivative(Ht0,Ht)
                    magn = list(map(lambda d: np.sum(np.absolute(d)),df))
                    # stop measures nonzeros in differention
                    if sum(magn) != 0:
                        df = list(map(lambda d,base:step*d/base,df,magn))
                        df = list(map(lambda i:df[i] +Lambda[i]+(ZL[i]-Zeta[i])/mu,range(3)))
                        # Gradient descent
                        ZL = list(map(lambda L,d:L - d,ZL,df))
                        # SVD: low rank
                        ZL = list(map(lambda L:SVD_Threshold(L,step*lam),ZL))

                        # Gradient descent
                        ZD = list(map(lambda D,d:D - np.diag(np.diag(d)),ZD,df))
                        # EVD: sparse
                        ZD = list(map(lambda D:Max_func(D - step*lam*np.identity(D.shape[0])),ZD))
                        # The last line is the same as SVD:low rank, but because of the difference of python function, they look different
                        Z = list(map(lambda D,L: D+L,ZD,ZL))
                        pickle.dump(Z,open(Z_name,"wb"))
                        Ht = self.cov_mat(Z[0],Z[1],Z[2])
                    else:
                        break
                    # Stop condition for GD
                    s = list(map(lambda i: np.linalg.norm(lastGD[i] - Z[i]),range(3)))
                    s = sum(s)
                    #if it2%30 == 0:
                    #    print("GD: s = %lf"%s)
                    if s <= eps:
                        break
                GD_it.append(GD_it_temp)  #*********analysis*********
                GD_t_temp = time.time() - GD_t_temp  #*********analysis*********
                GD_t.append(GD_t_temp)  #*********analysis*********
                # Stop condition for majorization
                s = list(map(lambda i: np.linalg.norm(lastMaj[i] - Z[i]),range(3)))
                s = sum(s)
                print("Maj: s = %lf"%s)
                if s <= eps:
                    break
            Maj_it.append(Maj_it_temp) #*********analysis*********
            Maj_t_temp = time.time() - Maj_t_temp #*********analysis*********
            Maj_t.append(Maj_t_temp) #*********analysis*********
            Theta_t_temp = time.time() - Theta_t_temp #*********analysis*********
            Theta_t.append(Theta_t_temp) #*********analysis*********
            # 3. Lambda step (level 3)
            Lambda_t_temp = time.time()#*********analysis*********
            for i in range(3):
                Lambda[i] = Lambda[i] - (1/mu)*(Zeta[i] - Z[i])
            Lambda_t_temp = time.time() - Lambda_t_temp #*********analysis*********
            Lambda_t.append(Lambda_t_temp)#*********analysis*********
        # ADMM termination condition
            r = list(map(lambda i: np.linalg.norm(ZL[i] - Zeta[i]),[0,1,2]))
            r = sum(r)
            s = list(map(lambda i: np.linalg.norm(Z[i] - lastADMM[i]),[0,1,2]))
            s = sum(s)
            print("ADMM: s = %lf"%s)
            if r <=eps or s <=eps:
                break

        ADMM_it.append(ADMM_it_temp) #*********analysis*********
        Total_t = time.time() - Total_t #*********analysis*********
        # Pie graph: 1
        Maj_t = sum(Maj_t)
        GD_t = sum(GD_t)
        # Pie graph: 2
        Zeta_t = np.mean(Zeta_t)
        Theta_t = np.mean(Theta_t)
        Lambda_t = np.mean(Lambda_t)
        #it numbers:
        ADMM_it = np.mean(ADMM_it)
        Maj_it = np.mean(Maj_it)
        GD_it = np.mean(GD_it)
        analysis = [Total_t,Maj_t,GD_t,Zeta_t,Theta_t,Lambda_t,ADMM_it,Maj_it,GD_it]
        T,N = (self.T,self.N)
        for dgpno in [1,2,3,4]:
            analysisname = Father_loc + "Program Analysis//analysis_lam = %lf_VEC%d(%d,%d).txt"%(lam,dgpno,T,N) # User self define
            if not os.path.exists(analysisname):
                break
        pickle.dump(analysis,open(analysisname,"wb"))

        return Ht.T
    def tunning(self):
        Min,Max = (-3,7)
        lam1 = np.linspace(Min,Max,20)
        lam1 = pow(10,lam1)
        T,N = (self.T,self.N)
        def temp(lam):
            try:
                return self.estimation(lam)
            except:
                print("Singular in lam = %d"%lam)
                return np.nan*np.ones((T,N,N))
        res1 = Parallel(n_jobs=20)(delayed(temp)(i) for i in lam1)
        if not os.path.exists(Father_loc+"tunning//"):
            os.makedirs(Father_loc+"tunning//")
        for dgpno in [1,2,3,4]:
            if not os.path.exists(Father_loc+"tunning//""res1_(%d,%d)_%d.txt"%(self.N,self.T,dgpno)):
                signature = dgpno
                break

        pickle.dump(res1,open(Father_loc+"tunning//""res1_(%d,%d)_%d.txt"%(self.N,self.T,signature),"wb"))
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
        pickle.dump(lam1,open(Father_loc+"tunning//lam1_(%d,%d)_%dc.txt"%(self.N,self.T,signature),"wb"))
        return res1[min_no]
