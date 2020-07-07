import numpy as np
import pandas as pd
import scipy.linalg
import importlib.util
import pickle
print("function list")
print("getRHt(X): transfer r Ht to python Ht")
print("Norm(H,H_hat)")
print("norm_name")
norm_name = [ 'fro','nuc', 'inf','-inf','1','-1','2','-2',"entropy loss","quadratic loss"]
print(norm_name)
print("entropy_loss(H,H)hat)")
print("quadratic_loss(H,H)hat)")
def getRHt(X):
    X = np.asmatrix(X)
    N,T = X.shape
    X = X[:,range(1,T)]
    T = int(T/N)
    Ht = np.zeros((T,N,N))
    for t in range(T):
        Ht[t,:,:] = X[:,range(t,t+N)]
    return Ht
def Norm(H,H_hat):
    # None, 'fro','nuc', inf,-inf,0,1,-1,2,-2
    order_list = [ 'fro','nuc', np.inf,-np.inf,1,-1,2,-2]
    H = H.T
    H_hat = H_hat.T
    N,N,T = H.shape
    norm = []
    for order in order_list:
        norm.append(np.mean(list(map(lambda t: np.linalg.norm(H[:,:,t] - H_hat[:,:,t],ord = order),range(T))))/(N**2))
    norm.append(np.mean(list(map(lambda t: entropy_loss(H[:,:,t],H_hat[:,:,t]),range(T))))/(N**2))
    norm.append(np.mean(list(map(lambda t: quadratic_loss(H[:,:,t],H_hat[:,:,t]),range(T))))/(N**2))
    return norm
def entropy_loss(H,H_hat):
    N,N = H.shape
    p = N
    mult = np.linalg.inv(H)@H_hat
    loss = np.matrix.trace(mult) - np.log(np.linalg.det(mult)) - p
    return loss
def quadratic_loss(H,H_hat):
    N,N = H.shape
    p = N
    mult = np.linalg.inv(H)@H_hat
    loss = np.matrix.trace(mult - np.identity(N))**2
    return loss
