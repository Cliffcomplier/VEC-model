import numpy as np
import pandas as pd
import scipy.linalg
import importlib.util
import pickle
import os
from datetime import date
txt = "D://Research//"
loc = '%s//General Code file//'%txt # User defined
res_loc = "%s//Thesis%s//"%(txt,date.today()) # User defined
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
if not os.path.exists(res_loc+"DGP//"):
    os.makedirs(res_loc+"DGP//")
if not os.path.exists(res_loc+"Result//"):
    os.makedirs(res_loc+"Result//")
if not os.path.exists(res_loc+"Norm//"):
    os.makedirs(res_loc+"Norm//")
# Data Generation
def Data_Generation(dgp,T,N):
    Obj = DGPs.DGPs(N,T,dgp)
    pickle.dump(Obj.epsi,open(res_loc + "DGP//VEC%d(%d,%d)_epsi.txt"%(dgp,T,N),"wb"))
    pickle.dump(Obj.Ht,open(res_loc + "DGP//VEC%d(%d,%d)_Ht.txt"%(dgp,T,N),"wb"))


def Simulation_tuning(dgp,T,N):
    epsi = pickle.load(open(res_loc + "DGP//VEC%d(%d,%d)_epsi.txt"%(dgp,T,N),"rb"))
    Ht = pickle.load(open(res_loc + "DGP//VEC%d(%d,%d)_Ht.txt"%(dgp,T,N),"rb"))
    Ht_hat = lr.VEC(epsi).tunning()
    norm = tool.Norm(Ht,Ht_hat)
    pickle.dump(Ht_hat,open(res_loc + "Result//VEC%d(%d,%d)_Ht_hat.txt"%(dgp,T,N),"wb"))
    pickle.dump(norm,open(res_loc + "Norm//VEC%d(%d,%d)_Norm.txt"%(dgp,T,N),"wb"))
    print("VEC%d(%d,%d) finished!"%(dgp,T,N))
def Simulation_estimation(dgp,T,N):
    epsi = pickle.load(open(res_loc + "DGP//VEC%d(%d,%d)_epsi.txt"%(dgp,T,N),"rb"))
    Ht = pickle.load(open(res_loc + "DGP//VEC%d(%d,%d)_Ht.txt"%(dgp,T,N),"rb"))
    Ht_hat = lr.VEC(epsi).estimation(0.1)

'''
spec = importlib.util.spec_from_file_location("Sim ", loc+"20191120.py")
Sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Sim)
Sim.Data_Generation(3,10,3)
Sim.Simulation_tuning(3,10,3)
'''
