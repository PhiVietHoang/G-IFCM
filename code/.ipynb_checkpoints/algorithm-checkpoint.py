import numpy as np

def normalization(R, a, b):
    Nd = np.zeros_like(R, dtype=float) 
    for i in range(R.shape[1]):
        min_val = np.min(R[:, i])
        max_val = np.max(R[:, i])
        Nd[:, i] = a + ((R[:, i] - min_val) / (max_val - min_val)) * (b - a)
    return Nd

def distance(Nd):
    P = Nd.shape[0]
    dis = np.zeros((P,P))
    for i in range(P):
        for j in range(P):
            dis[i, j] = np.linalg.norm(Nd[i] - Nd[j])
    
    return dis

def product(distances, Nd):
    prod = np.zeros_like(Nd, dtype=float)
    prod = np.matmul(distances,Nd)
    return prod

def membershipValue(prod):
    M = np.zeros_like(prod, dtype = float)
    M = 1/prod
    return M

def nonMembershipValue(M,alpha):
    N = np.zeros_like(M, dtype = float)
    N = (1-M**alpha)**(1/alpha)
    return N

def hesitancyValue(N,alpha):
    H = np.zeros_like(N, dtype = float)
    H = 1-N-(1-N**alpha)**(1/alpha)
    return H

def method1(R, alpha, a=0, b=1):
    Nd = normalization(R, a, b)
    dis = distance(Nd)
    prod = product(dis, Nd)
    M = membershipValue(prod)
    N = nonMembershipValue(M, alpha)
    H = hesitancyValue(N, alpha)
    return M, N, H

def method2(R, a, b, alpha, beta):
    Nd = normalization(R, a, b)
    dis = distance(Nd)
    prod = product(dis, Nd)
    factor = membershipValue(prod)
    sum = R + factor
    M = (normalization(sum,a,b))**beta
    N = (1-M**(alpha*beta))**(1/alpha)
    H = 1 - M - N
    return M, N, H

def method2_1(R, a, b, alpha, beta):
    Nd = normalization(R, a, b)
    prod1 = R+Nd
    Nd1 = normalization(prod1, a,b)
    dis = distance(Nd1)
    prod2 = product(dis, Nd1)
    factor = membershipValue(prod2)
    sum = R + factor
    M = (normalization(sum,a,b))**beta
    N = (1-M**(alpha*beta))**(1/alpha)
    H = 1 - M - N
    return M, N, H
  
def method3(R, c, m, alpha, beta=1, epsilon=1e-6, a=0, b=1):
    M, N, H = method1(R, alpha)
    X = np.dstack((M, N, H))
    d = X.shape[1]
    M_S = np.random.rand(c, d)
    N_S = nonMembershipValue(M_S, alpha)
    H_S = hesitancyValue(M_S, alpha)
    S_init = np.dstack((M_S, N_S, H_S))
    print('Initial centroids:')
    print(S_init)
    P = X.shape[0]
    iter_num = 0
    
    while 1:
        print(f'iteration: {iter_num}')
        U = np.zeros((P, c))
        for i in range(P):
            D = np.zeros(c)
            for k in range(c):
                D[k] = 1/(2*P)*distance_func(X[i], S_init[k])**(1/(m-1))
            
            D=D**(-1)
            for l in range(c):
                U[i,l] = D[l]/np.sum(D)
        
        print('U:')
        print(U)

        S = np.zeros_like(S_init)
        for l in range(c):
            mu_l = np.sum(np.array([(U[i,l]**m)*M[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            nu_l = np.sum(np.array([(U[i,l]**m)*N[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            pi_l = np.sum(np.array([(U[i,l]**m)*H[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            S[l] = np.dstack((mu_l, nu_l, pi_l))
        
        print('S:')
        print(S)
        
        criteria = 0
        for l in range(c):
            criteria += ((1/(2*P)*distance_func(S_init[l], S[l]))**(1/2))/c
        
        print(f'\nNorm diff: {criteria}\n')
        if criteria < epsilon: break
        
        iter_num += 1
        S_init = S
    
    return U, S
        
def distance_func(x, y):
    return np.sum((x-y)**2)