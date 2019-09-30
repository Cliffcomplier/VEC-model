class BEKK_dgp:
    def __init__(self,pars,N,T):
        A = np.asmatrix(np.reshape(pars[0:N*N],(N,N)))
        B = np.asmatrix(np.reshape(pars[N*N:2*N*N],(N,N)))
        C = np.asmatrix(np.zeros((N,N)))
        C[np.triu_indices(N)] = pars[2*N*N:len(pars)]
        Ht = np.zeros((N,N,T))
        epsi = np.zeros((N,T))
        Ht[:,:,0] =C@C.T + np.identity(N)
        epsi[:,0] = np.random.normal(0,1,N)@np.asmatrix(np.linalg.inv(scipy.linalg.sqrtm(Ht[:,:,0])))
        for t in range(1,T):
            eps = np.asmatrix(epsi[:,t-1]).T
            H = np.asmatrix(Ht[:,:,t-1])
            Ht[:,:,t] = C@C.T + A@eps@eps.T@A.T + B@H@B.T
            epsi[:,t] = np.random.normal(0,1,N)@np.asmatrix(np.linalg.inv(scipy.linalg.sqrtm(Ht[:,:,t])))
        self.epsi = epsi
        self.Ht = Ht
class Bekk:
    def __init__(self,eps):
        self.epsi = eps
        N,T = eps.shape
        A = 0.1*np.asmatrix(np.identity(N))
        B = 3*A
        C = np.asmatrix(np.triu(0.1*np.ones((N,N))))
        self.par0 = np.array(list(A.A1)+list(B.A1)+list(C[np.triu_indices(N)].A1))
        self.coef = scipy.optimize.minimize(self.loglik, self.par0, method='BFGS', jac=self.df,options={'disp': True}).x
    def loglik(self,pars):
        N,T = self.epsi.shape
        pars = np.asmatrix(pars).A1
        A = np.asmatrix(np.reshape(pars[0:N*N],(N,N)))
        B = np.asmatrix(np.reshape(pars[N*N:2*N*N],(N,N)))
        C = np.asmatrix(np.zeros((N,N)))
        C[np.triu_indices(N)] = pars[2*N*N:len(pars)]
        Ht = np.zeros((N,N,T))
        Ht[:,:,0] =C@C.T + np.identity(N)
        L = 0
        for t in range(1,T):
            eps = np.asmatrix(self.epsi[:,t-1]).T
            H = np.asmatrix(Ht[:,:,t-1])
            Ht[:,:,t] = C@C.T + A@eps@eps.T@A.T + B@H@B.T
            L = L - 0.5*N*np.log(2*math.pi) - 0.5*np.log(np.linalg.det(H)) - 0.5*eps.T@np.linalg.inv(H)@eps
            print(-L[0,0])
        return -L[0,0]
    def df(self,pars):
        N,T = self.epsi.shape
        pars = np.asmatrix(pars).A1
        A = np.asmatrix(np.reshape(pars[0:N*N],(N,N)))
        B = np.asmatrix(np.reshape(pars[N*N:2*N*N],(N,N)))
        C = np.asmatrix(np.zeros((N,N)))
        C[np.triu_indices(N)] = pars[2*N*N:len(pars)]
        da = np.asmatrix(np.zeros((N,N)))
        db = np.asmatrix(np.zeros((N,N)))
        dc = np.zeros((N,N))
        Ht = np.zeros((N,N,T))
        Ht[:,:,0] =C@C.T + np.identity(N)
        for t in range(1,T):
            eps = np.asmatrix(self.epsi[:,t-1]).T
            H = np.asmatrix(Ht[:,:,t-1])
            Ht[:,:,t] = C@C.T + A@eps@eps.T@A.T + B@H@B.T
            invH = np.linalg.inv(Ht[:,:,t])
            for i in range(N):
                for j in range(N):
                    pa = np.zeros((N,N))
                    pa[i,j] = 1
                    dha = pa@eps@eps.T@A.T + A@eps@eps.T@pa.T
                    da[i,j] = da[i,j] + -0.5*np.matrix.trace(invH@dha)[0,0] + 0.5*eps.T@invH@dha@invH@eps
                    dhb = pa@H@B.T + B@H@pa.T
                    db[i,j] = db[i,j] + -0.5*np.matrix.trace(invH@dhb)[0,0] + 0.5*eps.T@invH@dhb@invH@eps
                    if j>=i:
                        dhc = pa@C.T + C@pa.T
                        dc[i,j] = dc[i,j] + -0.5*np.matrix.trace(invH@dhc)[0,0] + 0.5*eps.T@invH@dhc@invH@eps
        da = np.multiply(da,np.identity(N))
        db = np.multiply(db,np.identity(N))
        dc = np.triu(dc)
        return -np.array(list(da.A1)+list(db.A1)+list(dc[np.triu_indices(N)]))/1e5
    def DFP(self):
        x0 = self.par0
        maxk = 1e5
        rho = 0.05
        sigma = 0.4
        epsilon = 1e-5 
        k = 0
        N = np.shape(x0)[0]
        Hk = np.identity(N)
        while k < maxk:
            gk = self.df(x0)
            if np.linalg.norm(gk) < epsilon:
                break
            dk = -1.0*np.dot(Hk,gk)

            m = 0;
            mk = 0
            while m < 20:
                if self.loglik(x0 + rho**m*dk) < self.loglik(x0) + sigma*rho**m*np.dot(gk,dk):
                    mk = m
                    break
                m += 1
 
            x = x0 + rho**mk*dk
            print ("第"+str(k)+"次的迭代结果为："+str(x))
            sk = x - x0
            yk = self.df(x) - gk

            if np.dot(sk,yk) > 0:
                Hy = np.dot(Hk,yk)
                sy = np.dot(sk,yk) 
                yHy = np.dot(np.dot(yk,Hk),yk) 
                Hk = Hk - 1.0*Hy.reshape((N,1))*Hy/yHy + 1.0*sk.reshape((N,1))*sk/sy

            k += 1
            x0 = x
        return x0
