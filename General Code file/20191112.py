import numpy as np
import pandas as pd
import scipy.linalg
import importlib.util
import pickle
import os
loc = 'Z://Cliff//General Code file//'
spec = importlib.util.spec_from_file_location("DGPs ", loc+"DGPs.py")
DGPs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DGPs )
spec = importlib.util.spec_from_file_location("lr ", loc+"VEC_lr.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr )
spec = importlib.util.spec_from_file_location("prec ", loc+"VEC_prec.py")
prec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prec)
spec = importlib.util.spec_from_file_location("tool ", loc+"tool.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
import os
os.chdir("Z://Cliff//")
filename = "20191112"
if not os.path.exists(filename):
    os.makedirs(filename)
for i in [2]:
    for N in [100]:
        print("i = %d and N = %d"%(i,N))
        T = 100
        print("prec VEC%d(%d,%d)"%(i,N,T))
        Obj = DGPs.DGPs(N,T,i)
        Ht = Obj.Ht
        epsi = Obj.epsi
        pickle.dump(Ht,open(filename+"//VEC%d(%d,%d)_Ht.txt"%(i,N,T),"wb"))
        pickle.dump(epsi,open(filename+"//VEC%d(%d,%d)_epsi.txt"%(i,N,T),"wb"))     
        Ht_hat = prec.VEC(epsi).tunning("Z://Cliff//20191112")
        pickle.dump(Ht_hat,open(filename+"//VEC%d(%d,%d)_Ht_hat.txt"%(i,N,T),"wb"))