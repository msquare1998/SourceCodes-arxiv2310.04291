import numpy as np
import tensorcircuit as tc
from numpy import kron, exp
import jax.numpy as jnp

tc.set_backend("jax")
tc.set_dtype("complex128")

"""███████████████████████████████████████████████████████████████████████████████████
          @ Initial state and Lattice info
███████████████████████████████████████████████████████████████████████████████████"""
def makeIniVec(nQ):
    circ = tc.Circuit(nQ)
    circ.x(range(nQ))
    circ.h(range(nQ))
    return circ.state()

linkJ = np.array([
    [4, 5], [5, 6], [6, 7], [7, 4],
    [12, 13], [13, 14], [14, 15], [15, 12]
])

linkK = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [8, 9], [9, 10], [10, 11], [11, 8],
    [0, 4], [2, 6], [5, 9], [7, 11],
    [8, 12], [10, 14], [13, 1], [15, 3]
])

linkKp = np.array([
    [1, 5], [3, 7], [4, 8], [6, 10],
    [9, 13], [11, 15], [12, 0], [14, 2]
])

edges = np.array([
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15]
])

""""███████████████████████████████████████████████████████████████████████████████████
          @ Make H0 (H_sp) and H1 (H_aux)
███████████████████████████████████████████████████████████████████████████████████"""
def makeH_sp(Lx, Ly, J, K, Kp):
    nQ = int(Lx * Ly)
    H_dg = np.zeros([2 ** nQ])

    diagZ = np.array([1, -1])
    diagI = np.array([1, 1])

    for i in range(len(linkJ)):
        _h_ = 1
        for j in range(nQ):
            if j in linkJ[i]:
                _h_ = kron(diagZ, _h_)
            else:
                _h_ = kron(diagI, _h_)
        H_dg += J * _h_

    for i in range(len(linkK)):
        _h_ = 1
        for j in range(nQ):
            if j in linkK[i]:
                _h_ = kron(diagZ, _h_)
            else:
                _h_ = kron(diagI, _h_)
        H_dg += K * _h_

    for i in range(len(linkKp)):
        _h_ = 1
        for j in range(nQ):
            if j in linkKp[i]:
                _h_ = kron(diagZ, _h_)
            else:
                _h_ = kron(diagI, _h_)
        H_dg += Kp * _h_

    H_sp_indices = np.array([[i, i] for i in range(2 ** nQ)])
    h = tc.backend.coo_sparse_matrix(indices=H_sp_indices, values=H_dg, shape=[2 ** nQ, 2 ** nQ])

    e0 = np.min(H_dg)
    e1 = 1e4
    for i in range(2 ** nQ):
        if H_dg[i] < e1 and H_dg[i] != e0:
            e1 = H_dg[i]
    return h, H_dg, e0, e1

def makeH_aux(nQ: int) -> np.ndarray:
    pauliX = np.array([[0, 1],
                       [1, 0]])

    pauliI = np.array([[1, 0],
                       [0, 1]])

    H = np.zeros((2 ** nQ, 2 ** nQ))

    for i in range(nQ):
        h = 1
        for q in range(nQ):
            if q == i:
                h = kron(pauliX, h)
            else:
                h = kron(pauliI, h)
        H += h
    return H

""""███████████████████████████████████████████████████████████████████████████████████
          @ Make H_edge
███████████████████████████████████████████████████████████████████████████████████"""
def makeH_edge(nQ: int, whichEdge: int):
    edgeList = edges[whichEdge]
    print(edgeList)
    pauliX = np.array([[0, 1],
                       [1, 0]])

    pauliI = np.array([[1, 0],
                       [0, 1]])

    H = np.zeros((2 ** nQ, 2 ** nQ))

    for e in edgeList:
        h = 1
        for q in range(nQ):
            if q == e:
                h = kron(pauliX, h)
            else:
                h = kron(pauliI, h)
        H += h
    return H

""""███████████████████████████████████████████████████████████████████████████████████
          @ Make diagonal evolution operator
███████████████████████████████████████████████████████████████████████████████████"""

def makeDiagEvoOp(nQ, h, dt):
    op = np.zeros([2 ** nQ]).astype(complex)
    for i in range(2 ** nQ):
        op[i] = exp(-1j * h[i] * dt)
    return jnp.array(op)


""""███████████████████████████████████████████████████████████████████████████████████
          @ Saving data and loading data, timer
███████████████████████████████████████████████████████████████████████████████████"""
# -------------------------------------------------------------------------------------------------------
#   Data saving, loading, and Timer
# -------------------------------------------------------------------------------------------------------
def saveState(vec, nQ, ds, dt, J, K, Kp):
    f_r = open(f'./stateVecInfo/realPart_ds{ds}_dt{dt}_J{J}_K{K}_K{Kp}.dat', 'w', encoding='utf-8')
    f_i = open(f'./stateVecInfo/imagPart_ds{ds}_dt{dt}_J{J}_K{K}_K{Kp}.dat', 'w', encoding='utf-8')
    for i in range(2 ** nQ):
        f_r.write(str(vec[i].real) + '\n')
        f_i.write(str(vec[i].imag) + '\n')
    f_r.close()
    f_i.close()

def saveList(path, theList):
    f0 = open(f'{path}', 'w', encoding='utf-8')
    for e in theList:
        f0.write(str(e) + '\n')
    f0.close()

def loadList(path):
    f0 = open(f'{path}', 'r', encoding='utf-8')
    theList = []
    idx = 0
    while True:
        line0 = f0.readline()
        if line0:
            theList.append(float(line0.strip('\n')))
            idx += 1
        else:
            break
    f0.close()
    return theList

def reportTimeUsed(t0, t1):
    dt = t1 - t0
    h = int(dt / 3600)
    m = int((dt - h * 3600) / 60)
    s = dt - 3600 * h - 60 * m
    print("Time used: %dh, %dm, %.2fs" % (h, m, s))

@tc.backend.jit
def calcEnergy(vec, h_sp):
    return (tc.backend.conj(tc.backend.transpose(vec)) @ h_sp @ vec).real