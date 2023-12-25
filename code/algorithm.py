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
  
def method3(R, c, m, alpha, ruler, beta=1, epsilon=1e-6, a=0, b=1):
    distance_functions = {
        'distance_function': distance_function,
        'hamming_distance': hamming_distance,
        'euclidean_distance': euclidean_distance,
        'normalized_euclidean_distance': normalized_euclidean_distance,
        'hausdorff_distance': hausdorff_distance,
        'yang_chiclana_distance': yang_chiclana_distance,
        'wang_xin_distance': wang_xin_distance,
        'liu_jiang_distance': liu_jiang_distance,
        'he_distance': he_distance,
        'thao_distance': thao_distance,
        'mahanta_panda_distance': mahanta_panda_distance,
    }
    
    distance_func = distance_functions.get(ruler)
    if not distance_func:
        raise ValueError(f'Unknown ruler: {ruler}')
    
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

def method3_1(R, c, m, alpha, ruler, beta=1, epsilon=1e-6, a=0, b=1):
    distance_functions = {
        'distance_function': distance_function,
        'hamming_distance': hamming_distance,
        'euclidean_distance': euclidean_distance,
        'normalized_euclidean_distance': normalized_euclidean_distance,
        'hausdorff_distance': hausdorff_distance,
        'yang_chiclana_distance': yang_chiclana_distance,
        'wang_xin_distance': wang_xin_distance,
        'liu_jiang_distance': liu_jiang_distance,
        'he_distance': he_distance,
        'thao_distance': thao_distance,
        'mahanta_panda_distance': mahanta_panda_distance,
    }
    
    distance_func = distance_functions.get(ruler)
    if not distance_func:
        raise ValueError(f'Unknown ruler: {ruler}')
    
    M, N, H = method2_1(R, a,b,alpha,beta)
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
        
def distance_function(x, y):
    return np.sum((x-y)**2)

def hamming_distance(x, y):
    return 0.5 * np.sum(np.abs(x - y))

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def normalized_euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2)) / np.sqrt(len(x))

def hausdorff_distance(x, y):
    return np.max(np.abs(x - y))

def yang_chiclana_distance(x, y):
    return np.max(np.abs(x - y))

def wang_xin_distance(x, y):
    abs_diff = np.abs(x - y)
    return (1/4) * np.sum(abs_diff) + (1/2) * np.max(abs_diff)

def liu_jiang_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def he_distance(x, y):
    abs_diff = np.abs(x - y)
    return 2 * np.sum(abs_diff) + np.sum(np.diff(abs_diff))

def thao_distance(x, y):
    return (1 / len(x)) * np.sum(np.abs(x[:, 0] - y[:, 0]) + np.abs(x[:, 1] - y[:, 1]) +
                                  np.abs(x[:, 0] - y[:, 0]) * np.abs(x[:, 1] - y[:, 1]))

def mahanta_panda_distance(x, y):
    abs_diff = np.abs(x - y)
    numerator = np.sum(abs_diff)
    denominator = np.sum(np.abs(x) + np.abs(y))
    return numerator / denominator if denominator != 0 else 0 


def calculate_CA(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def calculate_PC(U):
    return np.sum(U ** 2) / U.shape[0]

# def calculate_SC(X, U, centroids):
#     num = np.sum(U ** 2 * np.linalg.norm(X - centroids[:, np.newaxis], axis=2) ** 2)
#     den = np.sum(np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2) ** 2)
#     return num / den

# def calculate_XB(X, U, centroids):
#     num = np.sum(U ** 2 * np.linalg.norm(X - centroids[:, np.newaxis], axis=2) ** 2)
#     min_dist = np.min(np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2) ** 2 + np.eye(centroids.shape[0]))
#     return num / (X.shape[0] * min_dist)

# def calculate_DI(X, U, centroids):
#     distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2) + np.eye(centroids.shape[0]) * np.inf
#     min_intercluster_distance = np.min(distances)
#     max_intracluster_distance = max(np.max(np.linalg.norm(X - centroids[c], axis=1)) for c in range(centroids.shape[0]))
#     return min_intercluster_distance / max_intracluster_distance