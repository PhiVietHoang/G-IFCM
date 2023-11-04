import numpy as np

def normalization(R, a, b):
    Nd = np.zeros_like(R, dtype=float) 
    for i in range(R.shape[1]):
        min_val = np.min(R[:, i])
        max_val = np.max(R[:, i])
        Nd[:, i] = a + ((R[:, i] - min_val) / (max_val - min_val)) * (b - a)
    return Nd

def distance(Nd):  
    dis = np.zeros_like(Nd, dtype=float)
    for i in range(Nd.shape[0]):
        for j in range(Nd.shape[1]):
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

def method1(R, a, b, alpha):
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
    