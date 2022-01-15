""" modified from https://github.com/cdebacco/SpringRank
New version developed by Nicolò Ruggeri, Max Planck Institute for Intelligent Systems, Tuebingen, Germany, March-2020
It forces to use sparse matrices when possible, results in much more efficent implementation, especially for large matrices
"""

import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sparse

def build_from_dense(A, alpha, l0, l1):
    """
    Given as input a 2d numpy array, build the matrices A and B to feed to the linear system solver for SpringRank.
    """
    n = A.shape[0]
    k_in = np.sum(A, 0)
    k_out = np.sum(A, 1)

    D1 = k_in + k_out           # to be seen as diagonal matrix, stored as 1d array
    D2 = l1 * (k_out - k_in)    # to be seen as diagonal matrix, stored as 1d array

    if alpha != 0.:
        B = np.ones(n) * (alpha * l0) + D2
        A = - (A + A.T)
        A[np.arange(n), np.arange(n)] = alpha + D1 + np.diagonal(A)
    else:
        last_row_plus_col = (A[n - 1, :] + A[:, n - 1]).reshape((1, n))
        A = A + A.T
        A += last_row_plus_col

        A[np.arange(n), np.arange(n)] = A.diagonal() + D1
        D3 = np.ones(n) * (l1 * (k_out[n - 1] - k_in[n - 1]))  # to be seen as diagonal matrix, stored as 1d array
        B = D2 + D3

    return scipy.sparse.csr_matrix(A), B


def build_from_sparse(A, alpha, l0, l1):
    """
    Given as input a sparse 2d scipy array, build the matrices A and B to feed to the linear system solver for
    SpringRank.
    """
    n = A.shape[0]
    k_in = np.sum(A, 0).A1      # convert matrix of shape (1, n) into 1-dimensional array
    k_out = np.sum(A, 1).A1     # same with (n, 1) matrix

    D1 = k_in + k_out           # to be seen as diagonal matrix, stored as 1d array
    D2 = l1 * (k_out - k_in)    # to be seen as diagonal matrix, stored as 1d array

    if alpha != 0.:
        B = np.ones(n) * (alpha * l0) + D2
        A = - (A + A.T)
        # convert to lil matrix for more efficient computations
        A = A.tolil(copy=False)
        A.setdiag(alpha + D1 + A.diagonal())
    else:
        last_row_plus_col = sparse.COO.from_scipy_sparse(A[n - 1, :] + A[:, n - 1].T)   # create sparse 1d COO array
        A = A + A.T
        A += last_row_plus_col                                                          # broadcast on rows
        A = -A.tocsr()                                                                  # reconvert to csr scipy matrix

        # Notice that a scipy.sparse.SparseEfficiencyWarning will be raised by calling A.setdiag().
        # However converting to lil matrix with
        # A.tolil(copy=False)
        # is not computationally convenient. Just suppress the warning during the call of A.setdiag(...)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.sparse.SparseEfficiencyWarning)
            A.setdiag(A.diagonal() + D1)

        D3 = np.ones(n) * (l1 * (k_out[n-1] - k_in[n-1]))    # to be seen as diagonal matrix, stored as 1d array
        B = D2 + D3

    return A, B


def solve_linear_system(A, B, solver, verbose):
    if solver not in ['spsolve', 'bicgstab']:
        warnings.warn('Unknown parameter {solver} for argument solver. Setting solver = "bicgstab"'.format(solver=solver))
        solver = 'bicgstab'

    if verbose:
        print('Using scipy.sparse.linalg.{solver}(A,B)'.format(solver=solver))

    if solver == 'spsolve':
        sol = scipy.sparse.linalg.spsolve(A, B)
    elif solver == 'bicgstab':
        sol = scipy.sparse.linalg.bicgstab(A, B)[0]

    return sol.reshape((-1,))



def SpringRank(A, alpha=0., l0=1., l1=1., solver='bicgstab', verbose=False, force_dense=False):
    """
        Main routine to calculate SpringRank by a solving linear system.

        Parameters
        ----------
        A : numpy.ndarray or scipy.sparse.spmatrix
            Has tobe  2 dimensional and with same dimensions.
        alpha, l0, l1: float
            Defined as in the SpringRank paper
            https://arxiv.org/abs/1709.09002
        solver: str
            One between 'spsolve' (direct, slower) and 'bicgstab' (iterative, faster).
            The solver to be used for the linear system returning the ranks.
        verbose: bool
        force_dense: bool
            By default A is converted to a sparse matrix scipy.sparse.csr, if it is not already sparse.
            If force_dense is set to True and a dense ndarray A is input, then it is not converted to sparse.

        Returns
        -------
        rank
            numpy.ndarray of ranks. Indices represent the nodes' indices used in the matrix A.

    """

    # check if input is sparse or can be converted to sparse.
    use_sparse = True
    if force_dense and not scipy.sparse.issparse(A):
        try:
            A = scipy.sparse.csr_matrix(A)
        except:
            warnings.warn('The input parameter A could not be converted to scipy.sparse.csr_matrix. '
                          'Using a dense representation.')
            use_sparse = False
    elif force_dense:
        use_sparse = False

    # build array to feed linear system solver
    if use_sparse:
        A, B = build_from_sparse(A, alpha, l0, l1)
    else:
        A, B = build_from_dense(A, alpha, l0, l1)

    rank = solve_linear_system(A, B, solver, verbose)

    return rank


