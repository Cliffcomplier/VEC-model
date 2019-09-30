def vec(X):
    N,N = X.shape
    vech = np.ones(int(N*(N+1)/2))
    for i in range(0,N):
        for j in range(i,N):
            vech[i+j] = X[i,j]
    vech = np.asmatrix(vech).T
    return vech
def restack(vecx):
    Ns = len(vecx)
    N = int((np.sqrt(1+8*Ns) - 1)/2)
    X = np.asmatrix(np.reshape(np.ones(N*N),(N,N)))
    X[np.triu_indices(N)] = np.reshape(vecx,int((N*(N+1)/2),))
    X[np.tril_indices(N)] = np.reshape(vecx,int((N*(N+1)/2),))
    return X

def VEC_dgp(a,b,c,N,T):
    A = a*np.asmatrix(np.identity(int(N*(N+1)/2)))
    B = b*np.asmatrix(np.identity(int(N*(N+1)/2)))
    C = c*vec(np.identity(N))
    Ht = np.zeros((N,N,T))
    Ht[:,:,0] = np.identity(N)
    vecht = np.ones((int(N*(N+1)/2),T))
    epsi = np.asmatrix(np.zeros((N,T)))
    for t in range(1,T):
        d = C + A@vec(np.asmatrix(epsi[:,t-1])@np.asmatrix(epsi[:,t-1]).T)+ B@vec(np.asmatrix(Ht[:,:,t - 1]))
        vecht[:,t] = d.A1
        Ht[:,:,t] = restack(np.asmatrix(vecht[:,t]).T)
        epsi[:,t-1] = scipy.linalg.sqrtm(Ht[:,:,t])@np.random.normal(0,1,(N,1))
    return np.asmatrix(epsi)
def VEC(epsi):
    N,T = epsi.shape
    Ns = int(N*(N+1)/2)
    npar = int(Ns + 2*Ns**2)
    A = 0.5*np.asmatrix(np.identity(int(N*(N+1)/2)))
    B = 0.5*np.asmatrix(np.identity(int(N*(N+1)/2)))
    C = 0.5*vec(np.identity(N))
    par0 = np.array(list(A.A1)+list(B.A1)+list(C.A1))
    def VEC_loglik(pars,epsi = epsi):
        N = int((np.sqrt(1+4*(np.sqrt(1 + 8*len(pars)) - 1)/2) - 1)/2)
        A = pars[0:int(pow(N*(N+1)/2,2))]
        B = pars[int(pow(N*(N+1)/2,2)):2*int(pow(N*(N+1)/2,2))]
        C = pars[2*int(pow(N*(N+1)/2,2)):len(pars)]
        A = np.reshape(A,(int(N*(N+1)/2),int(N*(N+1)/2)))
        B = np.reshape(B,(int(N*(N+1)/2),int(N*(N+1)/2)))
        A = np.asmatrix(A)
        B = np.asmatrix(B)
        C = np.asmatrix(C).T
        N,T = epsi.shape
        Ht = np.ones((N,N,T))
        Ht[:,:,0] = np.identity(N)
        vecht = np.ones((int(N*(N+1)/2),T))
        L = -N*np.log(2*math.pi)/2 - np.log(np.linalg.det(Ht[:,:,0]))/2 - np.asmatrix(epsi[:,0]).T@np.linalg.inv(np.asmatrix(Ht[:,:,0]))@np.asmatrix(epsi[:,0])/2
        for t in range(1,T):
            d = vec(C)+ A@vec(np.asmatrix(epsi[:,t-1])@np.asmatrix(epsi[:,t-1]).T)+ B@vec(np.asmatrix(Ht[:,:,t - 1]))
            vecht[:,t] = d.A1
            Ht[:,:,t] = restack(np.asmatrix(vecht[:,t]).T)
            L = L + -N*np.log(2*math.pi)/2 - np.log(np.linalg.det(Ht[:,:,t]))/2 - np.asmatrix(epsi[:,t]).T@np.linalg.inv(np.asmatrix(Ht[:,:,t]))@np.asmatrix(epsi[:,t])/2
        print(-L[0,0])
        return -float(L)
    def VEC_df(pars,epsi = epsi):
        # Trans parameter vector into par matrix
        N,T = epsi.shape
        A = pars[0:int(pow(N*(N+1)/2,2))]
        B = pars[int(pow(N*(N+1)/2,2)):2*int(pow(N*(N+1)/2,2))]
        C = pars[2*int(pow(N*(N+1)/2,2)):len(pars)]
        A = np.reshape(A,(int(N*(N+1)/2),int(N*(N+1)/2)))
        B = np.reshape(B,(int(N*(N+1)/2),int(N*(N+1)/2)))
        A = np.asmatrix(A)
        B = np.asmatrix(B)
        C = np.asmatrix(C).T
        dc = np.asmatrix(np.zeros(int(N*(N+1)/2))).T
        da = np.asmatrix(np.zeros((int(N*(N+1)/2),int(N*(N+1)/2))))
        db = np.asmatrix(np.zeros((int(N*(N+1)/2),int(N*(N+1)/2))))
        Ht = np.zeros((N,N,T))
        Ht[:,:,0] = np.identity(N)
        vecht = np.asmatrix(np.zeros((int(N*(N+1)/2),T)))
        vecht[:,0] = vec(np.identity(N))
        Ns = int(N*(N+1)/2)
        da = np.zeros((Ns,Ns))
        db = np.zeros((Ns,Ns))
        dc = np.zeros((Ns,1))
        for t in range(1,T):
            H = C + A@vec(np.asmatrix(epsi[:,t-1])@np.asmatrix(epsi[:,t-1]).T)+ B@vec(np.asmatrix(Ht[:,:,t - 1]))
            vecht[:,t] = np.matrix(H.A1).T
            Ht[:,:,t] = restack(np.asmatrix(vecht[:,t]))
            invH = np.linalg.inv(Ht[:,:,t])
            veceps = vec(np.asmatrix(epsi[:,t])@np.asmatrix(epsi[:,t]).T)
            for i in range(N):
                for j in range(i,N):
                    if i==j:
                        dH = np.asmatrix(np.zeros((N,N)))
                        dH[i,j] = 1
                        dc[i+j] = dc[i+j] - 0.5*np.matrix.trace(invH@dH) + 0.5*np.asmatrix(epsi[:,t-1]).T@invH@dH@invH@np.asmatrix(epsi[:,t-1])
                        for k in range(Ns):
                            dH = np.asmatrix(np.zeros((N,N)))
                            dH[i,j] = veceps[k]
                            da[i+j,k] = da[i+j,k] - 0.5*np.matrix.trace(invH@dH) + 0.5*np.asmatrix(epsi[:,t-1]).T@invH@dH@invH@np.asmatrix(epsi[:,t-1])
                            dH[i,j] = vecht[k,t]
                            db[i+j,k] = db[i+j,k] - 0.5*np.matrix.trace(invH@dH) + 0.5*np.asmatrix(epsi[:,t-1]).T@invH@dH@invH@np.asmatrix(epsi[:,t-1])
                    else:
                        dH = np.asmatrix(np.zeros((N,N)))
                        dH[i,j] = 1
                        dc[i+j] = dc[i+j] - np.matrix.trace(invH@dH) + np.asmatrix(epsi[:,t-1]).T@invH@dH@invH@np.asmatrix(epsi[:,t-1])
                        for k in range(Ns):
                            dH = np.asmatrix(np.zeros((N,N)))
                            dH[i,j] = veceps[k]
                            da[i+j,k] = da[i+j,k] - np.matrix.trace(invH@dH) + np.asmatrix(epsi[:,t-1]).T@invH@dH@invH@np.asmatrix(epsi[:,t-1])
                            dH[i,j] = vecht[k,t]
                            db[i+j,k] = db[i+j,k] - np.matrix.trace(invH@dH) + np.asmatrix(epsi[:,t-1]).T@invH@dH@invH@np.asmatrix(epsi[:,t-1])
        da = np.asmatrix(da)
        db = np.asmatrix(db)
        dc = np.asmatrix(dc)
        return -np.array(list(da.A1)+list(db.A1)+list(dc.A1))
    VEC_loglik(par0,epsi)
    VEC_df(par0,epsi)
    res = scipy.optimize.minimize(VEC_loglik, par0, method='BFGS', jac=VEC_df,options={'disp': True})
    return res.x
#Example 
epsi = VEC_dgp(0.3,0.4,0.5,2,1000)
x = VEC(np.asmatrix(epsi))
