import numpy as np
from numpy.core.fromnumeric import transpose
import scipy.sparse as sp
from scipy.stats import rankdata
from scipy.sparse.linalg import svds
from scipy.linalg import orth
from sklearn.preprocessing import normalize

### convention: the larger the score/rank, the better

def syncRank(A):
    # update the meaning of a directed edge by tranpose, edited 20210725
    A = A.transpose()

    N = A.shape[1]

    # 1. Form C
    # Whenever Aij > Aji we set Cij = 1
    # Whenever Aij < Aji we set Cij = -1
    # Else, Cij = 0;
    # This means that Cij = sign(Aij - Aji)
    C = (A-A.transpose()).sign()

    # 2. Form Theta
    T = np.pi*C/(N-1)

    # 3. Form H
    H = sp.lil_matrix((N, N), dtype=complex)
    H[T!=0] = np.exp(1j*T[T!=0])

    # 4. Form Dinv
    Dinv = sp.diags(1./np.array((np.abs(H)).sum(axis=0)).reshape(-1))

    # 5. Form fancyH
    fancyH = Dinv.dot(H)

    # 6. Leading eigenvector of fancyH
    _, V = sp.linalg.eigs(fancyH,1,which='LM')

    # 7. Get angles in complex plane.
    angles = np.angle(V)

    # 8. Get order from angles
    idx = list(map(int,rankdata(angles)-1))
    sy = np.zeros(N)
    sy[idx] = np.arange(1, N+1)

    # 10. Choose the rank permutation that minimizes violations.
    viols = np.zeros(N)
    idx_perm = np.zeros(N)
    for ii in range(N):
        sy_perm = list(map(int,((sy + ii - 2) % N)))
        idx_perm[sy_perm] = np.arange(N)
        list_idx_perm = list(map(int, idx_perm))
        viols[ii] = (sp.triu(A[list_idx_perm][:,list_idx_perm])).sum()
    best = np.argmin(viols)

    sy = ((sy + best -2) % N) + 1
    
    return sy

def syncRank_angle(A):
    # update the meaning of a directed edge by tranpose, edited 20210725
    A = A.transpose()

    N = A.shape[1]

    # 1. Form C
    # Whenever Aij > Aji we set Cij = 1
    # Whenever Aij < Aji we set Cij = -1
    # Else, Cij = 0;
    # This means that Cij = sign(Aij - Aji)
    C = (A-A.transpose()).sign()

    # 2. Form Theta
    T = np.pi*C/(N-1)

    # 3. Form H
    H = sp.lil_matrix((N, N), dtype=complex)
    H[T!=0] = np.exp(1j*T[T!=0])

    # 4. Form Dinv
    Dinv = sp.diags(1./np.array((np.abs(H)).sum(axis=0)).reshape(-1))

    # 5. Form fancyH
    fancyH = Dinv.dot(H)

    # 6. Leading eigenvector of fancyH
    _, V = sp.linalg.eigs(fancyH,1,which='LM')

    # 7. Get angles in complex plane.
    angles = np.angle(V)

    # 8. Get order from angles
    idx = list(map(int,rankdata(angles)-1))
    sy = np.zeros(N)
    sy[idx] = np.arange(1, N+1)

    # 10. Choose the rank permutation that minimizes violations.
    viols = np.zeros(N)
    idx_perm = np.zeros(N)
    for ii in range(N):
        sy_perm = list(map(int,((sy + ii - 2) % N)))
        idx_perm[sy_perm] = np.arange(N)
        list_idx_perm = list(map(int, idx_perm))
        viols[ii] = (sp.triu(A[list_idx_perm][:,list_idx_perm])).sum()
    best = np.argmin(viols)
    
    return (angles - angles[best]).flatten()

def PageRank(A, d=0.8, v_quadratic_error=1e-12):
    A = A.toarray()
    N = max(A.shape) # N is equal to either dimension of M and the number of documents
    M = np.nan_to_num(A/A.sum(axis=0), nan=1/N)
    v = np.ones(N)
    v = v/np.linalg.norm(v, 1)   # This is now L1, not L2
    last_v = np.ones(N) * np.inf
    M_hat = (d * M) + (((1 - d) / N) * np.ones((N, N)))

    while(np.linalg.norm(v - last_v, 2) > v_quadratic_error):
        last_v = v
        v = M_hat.dot(v)
        v = v/np.linalg.norm(v, 1)
        # removed the L2 norm of the iterated PR

    return v

  
def eigenvectorCentrality(A,regularization=1e-6):
    # Eigenvector Centrality implemented below

    # Let's get the Perron Frobenius eigenvector of A
    A = sp.csr_matrix(A.toarray() + regularization)
    _, V = sp.linalg.eigs(A.asfptype(),1, which='LM')
    return V.flatten()

def rankCentrality(A):
    # In their text, a_ij = number of times j is preferred over i.
    # In the SpringRank paper, we usually assume the opposite. 
    # Here, we'll use the authors' direction, but note that whenever we call
    # this code, we'll have to transpose A. 
    
    # Note that there are no self-loops in this model, so we will check, 
    # discard, and warn 
    A = sp.lil_matrix(A.transpose()) # Q: True?
    A.setdiag(0)
    A = sp.csr_matrix(A)
    A.eliminate_zeros()
    
    
    # see Eq 5 of https://arxiv.org/pdf/1209.1688.pdf
    # We're going to regularize.
    # They suggest epsilon = 1. 
    # This seems extreme?
    
    # Not listed in the paper, but this is important. We have to regularize the
    # matrix A before we compute dmax. 
    regularization = 1
    A = sp.csr_matrix(A.toarray() + regularization)
    
    # Find dmax
    dout = A.sum(1)
    dmax = max(dout)
    
    # Eq 5
    
    P = sp.lil_matrix(A/(A+A.transpose())/dmax)
    
    # But we need to make sure that the matrix remains stochastic by making the
    # rows sum to 1. Without regularization, Eq 1 says P(i,i) = 1 - dout(i)/dmax;
    # Instead, we're going to just do this "manually"
    
    P = sp.csr_matrix(A)
    P.eliminate_zeros()
    D = sp.diags(np.array(1 - P.sum(1)).flatten())
    P = P + D

    
    _, V = sp.linalg.eigs(P.transpose(),1, which='LM')
    
    rc = V.flatten() / V.sum()
    
    return rc


# minimum violation ranking below
#   INPUTS:
# A is a NxN matrix representing a directed network
#   A can be weighted (integer or non-integer)
#   A(i,j) = # of dominance interactions by i toward j.
#   A(i,j) = # of times that j endorsed i.
# n_samples is an integer number of independent replicates of the MVR MCMC
# search procedure.
#   OUTPUTS:
# best_ranks is a vector of ranks. ONE IS BEST. N IS WORST
# best_violations is the number of violations
# best_A is the reordered matrix whose lower triangle contains min. viols.

def compute_violations_change(A,ii,jj):
    # Let's arbitrarily choose i to fall (larger rank number) and j to rise
    # (smaller rank number).
    i = min(ii,jj)
    j = max(ii,jj)
    dx= -A[j,i:j-1].sum() + A[i,i+1:j].sum() - A[i+1:j-1,i].sum() + A[i+1:j-1,j].sum()
    return dx

def compute_violations(B):
    x = sp.tril(B,-1).sum()
    return x

def mvr_single(A):
    violations = compute_violations(A)

    N = A.shape[0]
    order = np.arange(N)

    fails = 0
    hist_viols = [violations]
    hist_viols_backup = [violations]
    hist_fails = [fails]
    hist_swaps = []

    # RANDOM STEPS - Randomly swap till N^2 failures in a row.
    while True:
        i = np.random.randint(0, N, 1)[0] # choose random node
        j = np.random.randint(0, N, 1)[0] # choose second random node.
        while j==i: # make sure different
            i = np.random.randint(0, N, 1)[0]
            j = np.random.randint(0, N, 1)[0]
        dx = compute_violations_change(A,i,j)
        if dx < 0:
            order[[i,j]] = order[[j,i]]
            A[[i,j],:] = A[[j,i],:]
            A[:,[i,j]] = A[:,[j,i]]
            hist_swaps.append([i,j])
            hist_fails.append(fails)
            hist_viols.append(hist_viols[-1]+dx)
            violations = compute_violations(A)
            hist_viols_backup.append(violations)
            fails = 0
        else:
            fails += 1

        if fails == N*N:
            break

    # DETERMINISTIC STEPS - Find any local steps deterministically by search.
    counter = 0 # added by Yixuan He
    max_iter = 50 # added by Yixuan He
    while True:
        dxbest = 0
        for i in range(N-1):
            for j in range(i+1, N):
                dx = compute_violations_change(A,i,j)
                if dx < dxbest:
                    bestSwap = [i,j]
                    dxbest = dx
        if dxbest==0 or counter > max_iter:
            ranks = list(map(int,rankdata(order)-1))
            return ranks,violations,A
        else:
            counter += 1
        i = bestSwap[0]
        j = bestSwap[1]

        order[[i,j]] = order[[j,i]]

        A[[i,j],:] = A[[j,i],:]
        A[:,[i,j]] = A[:,[j,i]]

        hist_swaps.append([i,j])
        hist_viols.append(hist_viols[-1]+dxbest)
        violations = compute_violations(A)
        hist_viols_backup.append(violations)

def mvr(A_input,n_samples=5):

    A = A_input.copy() # not to modify the input adjacency matrix

    best_violations = np.power(A.shape[0], 2)
    best_ranks = None

    for _ in range(n_samples):
        ranks,violations,A = mvr_single(A)
        if violations < best_violations:
            best_violations = violations
            best_ranks = ranks
            best_A = A
    if best_ranks is None:
        best_ranks = ranks
    return np.array(best_ranks)


def serialRank(A):
    S = serialRank_matrix(A)
    L = sp.diags(np.array(S.sum(1)).flatten()) - S
    _, V = sp.linalg.eigs(L.asfptype(),2,which='SM')
    serr = np.real(V[:,1])

    return serr

def serialRank_matrix(A):
    # In serialRank, C(i,j) = 1 if j was preferred over i, so we need to transpose A.
    A = A.transpose()
    C = (A-A.transpose()).sign()
    n = A.shape[0]
    S = C.dot(C.transpose())/2
    S.data += n/2
    return S


def btl(A,tol=1e-3):
    # g = btl(A,tol)
    #   INPUTS:
    # A is a NxN matrix representing a directed network
    #   A can be weighted (integer or non-integer)
    #   A(i,j) = # of dominance interactions by i toward j. 
    #   A(i,j) = # of times that j endorsed i.
    # tol is the accuracy tolerance desired for successive iterations
    #   OUTPUTS:
    # s is the Nx1 vector of Davids Score
    #   Note: implementation of a regularized version (for dangling node)
    #   version of the algorithm presented in 
    # Hunter DR (2004) MM algorithms for generalized Bradley-Terry models. 
    # Annals of Statistics pp. 384?406

    A = sp.lil_matrix(A)
    A.setdiag(0)
    A = sp.csr_matrix(A)
    A.eliminate_zeros()
    N = A.shape[0]
    g = np.random.uniform(size=N) # random initial guesss
    wins = np.array(A.sum(1)).flatten()
    matches = A + A.transpose()
    totalMatches = np.array(matches.sum(0)).flatten()
    g_prev = np.random.uniform(size=N)
    eps=1e-6
    while np.linalg.norm(g-g_prev, 2) > tol:
        g_prev = g
        for i in range(N):
            if totalMatches[i]>0:
                q = matches[i].toarray().flatten()/(g_prev[i]+g_prev)
                q[i] = 0
                g[i] = (wins[i]+eps)/np.sum(q)
            else:
                g[i] = 0
        g = g/np.sum(g)
    return g

def davidScore(A):
    # s = davidScore(A)
    #   INPUTS:
    # A is a NxN matrix representing a directed network
    #   A can be weighted (integer or non-integer)
    #   A(i,j) = # of dominance interactions by i toward j. 
    #   A(i,j) = # of times that j endorsed i.
    #   OUTPUTS:
    # s is the Nx1 vector of Davids Score
    P = A/(A + A.transpose()) # Pij = Aij / (Aij + Aji)
    P = sp.lil_matrix(np.nan_to_num(P))
    P.setdiag(0)
    P = sp.csr_matrix(P)
    P.eliminate_zeros()
    w = P.sum(1)
    l = P.sum(0).transpose()
    w2 = P.dot(w)
    l2 = P.transpose().dot(l)
    s = w+w2-l-l2
    return np.array(s).flatten()

def SVD_RS(A):
    H = A - A.transpose()
    n = A.shape[1]
    u, s, vt = svds(H.asfptype(), k=2)
    u_orth = orth(u)
    u1 = np.ones((n,))/np.sqrt(n)
    u1_coeff = u1.dot(u_orth)
    u1_bar = u1_coeff[0] * u_orth[:, 0] + u1_coeff[1] * u_orth[:, 1]
    u1_bar = normalize(u1_bar[None, :], 'l2')
    u2 = u_orth[:,1]
    u2 = u2 - u1_bar.dot(u2) * u1_bar
    u2_bar = normalize(u2, norm='l2')
    e = np.ones_like(u2_bar)
    indices = (H != 0)
    T1 = np.matmul(np.transpose(u2_bar), e) - np.matmul(u2_bar, np.transpose(e))
    Pi = H[indices]/T1[indices.toarray()]
    tau = np.median(np.array(Pi))
    score = tau * u2_bar - tau * e.dot(np.transpose(u2_bar))/n * e
    return score.flatten()

def sqrtinvdiag(M):
    """Inverts and square-roots a positive diagonal matrix.
    Args:
        M (csc matrix): matrix to invert
    Returns:
        scipy sparse matrix of inverted square-root of diagonal
    """

    d = M.diagonal()
    dd = [1 / max(np.sqrt(x), 1 / 999999999) for x in d]

    return sp.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()
    
def SVD_NRS(A):
    H = A - A.transpose()
    n = A.shape[1]
    e = np.ones((1, n))
    D = sp.diags(np.array((np.abs(H)).sum(axis=0)).reshape(-1))
    D_sqrtinv = sqrtinvdiag(D)
    Hss = D_sqrtinv.dot(H).dot(D_sqrtinv)
    u, s, vt = svds(Hss.asfptype(), k=2)
    u_orth = orth(u)
    u1 = normalize(np.transpose(D_sqrtinv.dot(np.transpose(e)))).flatten()
    u1_coeff = u1.dot(u_orth)
    u1_bar = u1_coeff[0] * u_orth[:, 0] + u1_coeff[1] * u_orth[:, 1]
    u1_bar = normalize(u1_bar[None, :], 'l2')
    u2 = u_orth[:,1]
    u2 = u2 - u1_bar.dot(u2) * u1_bar
    u2_bar = normalize(u2, norm='l2')
    score = np.transpose(D_sqrtinv.dot(np.transpose(u2_bar)))
    indices = (H != 0)
    T1 = np.matmul(np.transpose(score), e) - np.matmul(score, np.transpose(e))
    Pi = H[indices]/T1[indices.toarray()]
    tau = np.median(np.array(Pi))
    score = tau * score - tau * e.dot(np.transpose(score))/n * e
    return score.flatten()