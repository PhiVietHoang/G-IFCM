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

def combined_distance(x, y, weights):
    distances = np.array([distance_function(x, y),
                          hamming_distance(x, y),
                          euclidean_distance(x, y),
                          normalized_euclidean_distance(x, y),
                          hausdorff_distance(x, y),
                          yang_chiclana_distance(x, y),
                          wang_xin_distance(x, y),
                          liu_jiang_distance(x, y),
                          he_distance(x, y),
                          thao_distance(x, y),
                          mahanta_panda_distance(x, y)])

    # Đặt trọng số cho từng độ đo
    weighted_distances = distances * weights

    # Tính trung bình có trọng số
    combined_distance = np.sum(weighted_distances) / np.sum(weights)
    
    return combined_distance

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
  
def method3(R, c, m, alpha,beta, ruler, epsilon=1e-6, a=0, b=1):
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
    # print('Initial centroids:')
    # print(S_init)
    P = X.shape[0]
    iter_num = 0
    
    while 1:
        # print(f'iteration: {iter_num}')
        U = np.zeros((P, c))
        for i in range(P):
            D = np.zeros(c)
            for k in range(c):
                D[k] = 1/(2*P)*distance_func(X[i], S_init[k])**(1/(m-1))
            
            D=D**(-1)
            for l in range(c):
                U[i,l] = D[l]/np.sum(D)
        
        # print('U:')
        # print(U)

        S = np.zeros_like(S_init)
        for l in range(c):
            mu_l = np.sum(np.array([(U[i,l]**m)*M[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            nu_l = np.sum(np.array([(U[i,l]**m)*N[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            pi_l = np.sum(np.array([(U[i,l]**m)*H[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            S[l] = np.dstack((mu_l, nu_l, pi_l))
        
        # print('S:')
        # print(S)
        
        criteria = 0
        for l in range(c):
            criteria += ((1/(2*P)*distance_func(S_init[l], S[l]))**(1/2))/c
        
        # print(f'\nNorm diff: {criteria}\n')
        if criteria < epsilon: break
        
        iter_num += 1
        S_init = S
    
    return U, S

def method3_1(R, c, m, alpha, beta, ruler, epsilon=1e-6, a=0, b=1):
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
    # print('Initial centroids:')
    # print(S_init)
    P = X.shape[0]
    iter_num = 0
    
    while 1:
        # print(f'iteration: {iter_num}')
        U = np.zeros((P, c))
        for i in range(P):
            D = np.zeros(c)
            for k in range(c):
                D[k] = 1/(2*P)*distance_func(X[i], S_init[k])**(1/(m-1))
            
            D=D**(-1)
            for l in range(c):
                U[i,l] = D[l]/np.sum(D)
        
        # print('U:')
        # print(U)

        S = np.zeros_like(S_init)
        for l in range(c):
            mu_l = np.sum(np.array([(U[i,l]**m)*M[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            nu_l = np.sum(np.array([(U[i,l]**m)*N[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            pi_l = np.sum(np.array([(U[i,l]**m)*H[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            S[l] = np.dstack((mu_l, nu_l, pi_l))
        
        # print('S:')
        # print(S)
        
        criteria = 0
        for l in range(c):
            criteria += ((1/(2*P)*distance_func(S_init[l], S[l]))**(1/2))/c
        
        # print(f'\nNorm diff: {criteria}\n')
        if criteria < epsilon: break
        
        iter_num += 1
        S_init = S
    
    return U, S

def method3_combined(R, c, m, alpha, beta, weights_array, epsilon=1e-6, a=0, b=1):
    # distance_functions = {
    #     'distance_function': distance_function,
    #     'hamming_distance': hamming_distance,
    #     'euclidean_distance': euclidean_distance,
    #     'normalized_euclidean_distance': normalized_euclidean_distance,
    #     'hausdorff_distance': hausdorff_distance,
    #     'yang_chiclana_distance': yang_chiclana_distance,
    #     'wang_xin_distance': wang_xin_distance,
    #     'liu_jiang_distance': liu_jiang_distance,
    #     'he_distance': he_distance,
    #     'thao_distance': thao_distance,
    #     'mahanta_panda_distance': mahanta_panda_distance,
    # }
    
    # distance_func = distance_functions.get(ruler)
    # if not distance_func:
    #     raise ValueError(f'Unknown ruler: {ruler}')
    
    M, N, H = method1(R, alpha)
    X = np.dstack((M, N, H))
    d = X.shape[1]
    M_S = np.random.rand(c, d)
    N_S = nonMembershipValue(M_S, alpha)
    H_S = hesitancyValue(M_S, alpha)
    S_init = np.dstack((M_S, N_S, H_S))
    # print('Initial centroids:')
    # print(S_init)
    P = X.shape[0]
    iter_num = 0
    
    while 1:
        # print(f'iteration: {iter_num}')
        U = np.zeros((P, c))
        for i in range(P):
            D = np.zeros(c)
            for k in range(c):
                D[k] = 1/(2*P)*combined_distance(X[i], S_init[k], weights_array)**(1/(m-1))
            
            D=D**(-1)
            for l in range(c):
                U[i,l] = D[l]/np.sum(D)
        
        # print('U:')
        # print(U)

        S = np.zeros_like(S_init)
        for l in range(c):
            mu_l = np.sum(np.array([(U[i,l]**m)*M[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            nu_l = np.sum(np.array([(U[i,l]**m)*N[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            pi_l = np.sum(np.array([(U[i,l]**m)*H[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            S[l] = np.dstack((mu_l, nu_l, pi_l))
        
        # print('S:')
        # print(S)
        
        criteria = 0
        for l in range(c):
            criteria += ((1/(2*P)*combined_distance(S_init[l], S[l], weights_array))**(1/2))/c
        
        # print(f'\nNorm diff: {criteria}\n')
        if criteria < epsilon: break
        
        iter_num += 1
        S_init = S
    
    return U, S

def method3_combined_method_2(R, c, m, alpha, beta, weights_array, epsilon=1e-6, a=0, b=1):
    # distance_functions = {
    #     'distance_function': distance_function,
    #     'hamming_distance': hamming_distance,
    #     'euclidean_distance': euclidean_distance,
    #     'normalized_euclidean_distance': normalized_euclidean_distance,
    #     'hausdorff_distance': hausdorff_distance,
    #     'yang_chiclana_distance': yang_chiclana_distance,
    #     'wang_xin_distance': wang_xin_distance,
    #     'liu_jiang_distance': liu_jiang_distance,
    #     'he_distance': he_distance,
    #     'thao_distance': thao_distance,
    #     'mahanta_panda_distance': mahanta_panda_distance,
    # }
    
    # distance_func = distance_functions.get(ruler)
    # if not distance_func:
    #     raise ValueError(f'Unknown ruler: {ruler}')
    
    M, N, H = method2_1(R, a,b,alpha,beta)
    X = np.dstack((M, N, H))
    d = X.shape[1]
    M_S = np.random.rand(c, d)
    N_S = nonMembershipValue(M_S, alpha)
    H_S = hesitancyValue(M_S, alpha)
    S_init = np.dstack((M_S, N_S, H_S))
    # print('Initial centroids:')
    # print(S_init)
    P = X.shape[0]
    iter_num = 0
    
    while 1:
        # print(f'iteration: {iter_num}')
        U = np.zeros((P, c))
        for i in range(P):
            D = np.zeros(c)
            for k in range(c):
                D[k] = 1/(2*P)*combined_distance(X[i], S_init[k], weights_array)**(1/(m-1))
            
            D=D**(-1)
            for l in range(c):
                U[i,l] = D[l]/np.sum(D)
        
        # print('U:')
        # print(U)

        S = np.zeros_like(S_init)
        for l in range(c):
            mu_l = np.sum(np.array([(U[i,l]**m)*M[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            nu_l = np.sum(np.array([(U[i,l]**m)*N[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            pi_l = np.sum(np.array([(U[i,l]**m)*H[i] for i in range(P)]), axis=0)/np.sum(U[:,l]**m)
            S[l] = np.dstack((mu_l, nu_l, pi_l))
        
        # print('S:')
        # print(S)
        
        criteria = 0
        for l in range(c):
            criteria += ((1/(2*P)*combined_distance(S_init[l], S[l], weights_array))**(1/2))/c
        
        # print(f'\nNorm diff: {criteria}\n')
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

def calculate_SC(X, U, centroids, m=2):
    # Số lượng điểm dữ liệu và cụm
    P, C = U.shape
    
    # Tính tử số của SC - tổng độ đậm đặc của các cụm
    numerator = 0.0
    for i in range(P):
        for j in range(C):
            numerator += U[i, j] ** m * np.linalg.norm(X[i] - centroids[j]) ** 2

    # Tính mẫu số của SC - tổng khoảng cách giữa các tâm cụm
    denominator = 0.0
    for k in range(P):
        for l in range(C):
            denominator += U[k, l] * np.sum([np.linalg.norm(centroids[l] - centroids[t]) ** 2 for t in range(C)])

    # Đảm bảo mẫu số không phải là zero để tránh lỗi chia cho zero
    if denominator == 0:
        return np.inf

    SC = numerator / denominator

    return SC

def calculate_XB(X, U, centroids):
    n = X.shape[0]
    c = centroids.shape[0]
    d = np.zeros((c,c))
    for i in range(c):
        for j in range(c):
            d[i,j] = np.linalg.norm(centroids[i]-centroids[j])**2
    
    sep = min(d[np.where(d > 0)])
    tmp = 0.0
    for i in range(c):
        tmp += sum(np.linalg.norm(X - centroids[i], axis=(1,2))*(U[:,i]**2))
    
    return tmp/n/sep

def calculate_DI(X, U):
    try:
        c = U.shape[1]
        n = X.shape[0]
        A = []
        for i in range(c):
            A.append([])
        
        ids = np.argmax(U, axis=1)
        for i in range(n):
            A[ids[i]].append(X[i])

        dias = np.zeros(c)
        for i in range(c):
            q = len(A[i])
            dia = np.zeros((q,q))
            for j in range(q):
                for k in range(q):
                    dia[j,k] = np.linalg.norm(A[i][j] - A[i][k])
            dias[i] = np.max(dia[np.where(dia > 0)])
            
        dia_max = np.max(dias)
        
        min_dis = np.linalg.norm(A[0][0] - A[1][0])
        for i in range(c):
            for j in range(i+1, c-i):
                q1 = len(A[i])
                q2 = len(A[j])
                dis = np.zeros((q1,q2))
                for i1 in range(q1):
                    for i2 in range(q2):
                        dis[i1, i2] = np.linalg.norm(A[i][i1] - A[j][i2])
                if min_dis > np.min(dis):
                    min_dis = np.min(dis)
        
        return min_dis/dia_max
    except Exception as e:
        print(f"Error calculating DI: {e}")
        return 0