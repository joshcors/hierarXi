# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# import numpy as np
cimport numpy as cnp
from cython.parallel import prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def reorder_out_of_place(float[:, :] src,      # (N, D) C-contig
                         unsigned int[:] perm,  # (N,)  perm[new] -> old
                         float[:, :] dst,       # (N, D) C-contig
                         int block = 65536):
    cdef Py_ssize_t N = src.shape[0], D = src.shape[1]
    cdef Py_ssize_t i, j, k, ii, jj, old_idx
    with nogil:
        for i in prange(0, N, block, schedule='static'):
            j = i + block
            if j > N: j = N
            # gather block
            for ii in range(i, j):
                old_idx = perm[ii]
                # memcpy one row
                for k in range(D):
                    dst[ii, k] = src[old_idx, k]

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_cyclic_reorder(cnp.uint16_t[:, :] points,
                         cnp.int32_t[:, :] cycle_fragments,
                         cnp.uint16_t[:, :] tails):
    
    cdef Py_ssize_t D = points.shape[1]
    cdef Py_ssize_t Ncyc = cycle_fragments.shape[0], cyc_size = cycle_fragments.shape[1]
    cdef Py_ssize_t i, j, k

    with nogil:
        for i in prange(0, Ncyc, schedule="static"):
            for j in range(cyc_size - 2):
                if cycle_fragments[i, j + 2] == -1:
                    for k in range(D):
                        points[cycle_fragments[i, j], k] = tails[i, k]
                    break
                else:
                    for k in range(D):
                        points[cycle_fragments[i, j], k] = points[cycle_fragments[i, j + 1], k]

            if cycle_fragments[i, cyc_size - 1] != -1:
                for k in range(D):
                    points[cycle_fragments[i, cyc_size - 2], k] = tails[i, k]
                
