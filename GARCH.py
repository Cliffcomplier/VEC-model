def GARCH(epsi,method):
    T = epsi.shape[0]
    L_list = []
    par_list = []
    for i in range(10):
        for j in range(3):
            par0[j] = random.choice(np.linspace(0,1,10))
        garch = scipy.optimize.minimize(GARCH_loglik, par0, method='BFGS', jac=GARCH_df,options={'disp': True})
        L = GARCH_loglik(epsi,pars = garch.x)
        L_list.append(L)
        par_list.append( garch.x)
    return (L_list,par_list)
