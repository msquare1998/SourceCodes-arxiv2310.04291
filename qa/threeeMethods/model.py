# ███████████████████████████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ Header for './threeMethods.py'
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Aug 15, 2023
# ███████████████████████████████████████████████████████████████████████████████████████████████████████
import numpy as np
from numpy import kron, exp
import tensorcircuit as tc

# -------------------------------------------------------------------------------------------------------
#   Make relevant Hamiltonians
# -------------------------------------------------------------------------------------------------------
def makeH_1D_Ising_PBC(nQ, J):
    H_dg = np.zeros([2 ** nQ])

    diagZ = np.array([1, -1])
    diagI = np.array([1, 1])

    links = np.array([[i, (i + 1) % nQ] for i in range(nQ)])

    for i in range(len(links)):
        _h_ = 1
        for j in range(nQ):
            if j in links[i]:
                _h_ = kron(diagZ, _h_)
            else:
                _h_ = kron(diagI, _h_)
        H_dg += J * _h_

    H_sp_indices = np.array([[i, i] for i in range(2 ** nQ)])
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    h = tc.backend.coo_sparse_matrix(indices=H_sp_indices, values=H_dg, shape=[2 ** nQ, 2 ** nQ])

    e0 = np.min(H_dg)
    e1 = 1e4
    for i in range(2 ** nQ):
        if H_dg[i] < e1 and H_dg[i] != e0:
            e1 = H_dg[i]

    return h, H_dg, e0, e1

def makeH_1D_Ising_OBC(nQ, J):
    H_dg = np.zeros([2 ** nQ])

    diagZ = np.array([1, -1])
    diagI = np.array([1, 1])

    links = np.array([[i, i + 1] for i in range(nQ - 1)])

    for i in range(len(links)):
        _h_ = 1
        for j in range(nQ):
            if j in links[i]:
                _h_ = kron(diagZ, _h_)
            else:
                _h_ = kron(diagI, _h_)
        H_dg += J * _h_

    H_sp_indices = np.array([[i, i] for i in range(2 ** nQ)])
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    h = tc.backend.coo_sparse_matrix(indices=H_sp_indices, values=H_dg, shape=[2 ** nQ, 2 ** nQ])

    e0 = np.min(H_dg)
    e1 = 1e4
    for i in range(2 ** nQ):
        if H_dg[i] < e1 and H_dg[i] != e0:
            e1 = H_dg[i]

    return h, H_dg, e0, e1

def makeH_aux(nQ):
    pauliX = np.array([
        [0, 1],
        [1, 0]
    ])

    pauliI = np.array([
        [1, 0],
        [0, 1]
    ])

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


# -------------------------------------------------------------------------------------------------------
#   Make the initial state
# -------------------------------------------------------------------------------------------------------
def makeIniVec_old(nQ):
    minusState = np.array([
        [1 / 2 ** 0.5],
        [-1 / 2 ** 0.5]])
    vec = 1
    for _ in range(nQ):
        vec = kron(vec, minusState)
    return vec

def makeIniVec(nQ):
    circ = tc.Circuit(nQ)
    circ.x(range(nQ))
    circ.h(range(nQ))
    return circ.state()

# -------------------------------------------------------------------------------------------------------
#   Calculating energy
# -------------------------------------------------------------------------------------------------------
def calcEnergy_old(H, stateVec):
    return (stateVec.conjugate().T @ H @ stateVec)[0][0].real

# -------------------------------------------------------------------------------------------------------
#   Modules for the 3rd method
# -------------------------------------------------------------------------------------------------------
def makeDiagEvoOp(nQ, h, dt):
    op = np.zeros([2 ** nQ]).astype(complex)
    for i in range(2 ** nQ):
        op[i] = exp(-1j * h[i] * dt)
    return op


# -------------------------------------------------------------------------------------------------------
#   Data saving, loading, and Timer
# -------------------------------------------------------------------------------------------------------
def saveE_list(fileName, E_list):
    f0 = open(f'./energyList/{fileName}.dat', 'w', encoding='utf-8')
    for e in E_list:
        f0.write(str(e) + '\n')
    f0.close()

def loadE_lsit(fileName):
    f0 = open(f'./energyList/{fileName}.dat', 'r', encoding='utf-8')
    E_list = []
    idx = 0
    while True:
        line0 = f0.readline()
        if line0:
            E_list.append(float(line0.strip('\n')))
            idx += 1
        else:
            break
    f0.close()
    return E_list

def reportTimeUsed(t0, t1):
    dt = t1 - t0
    h = int(dt / 3600)
    m = int((dt - h * 3600) / 60)
    s = dt - 3600 * h - 60 * m
    print("Time used: %dh, %dm, %.2fs" % (h, m, s))