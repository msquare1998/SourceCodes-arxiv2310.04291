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

# The same as the square lattice
edges = np.array([
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15]
])

""""███████████████████████████████████████████████████████████████████████████████████
          @ Make H0 (H_sp) and H1 (H_aux)
███████████████████████████████████████████████████████████████████████████████████"""
def makeLinkNNX_PBC(Lx, Ly):
    linkNNX_PBC = []
    for j in range(Ly):
        for i in range(Lx - 1):
            linkNNX_PBC.append([j * Lx + i, j * Lx + i + 1])
        linkNNX_PBC.append([j * Lx + Lx - 1, j * Lx])

    return np.array(linkNNX_PBC)

def makeLinkNNA_PBC(Lx, Ly):
    linkNNA_PBC = []
    # Not across boundary (y)
    for j in range(Ly - 1):
        if j % 2 == 0:
            linkNNA_PBC.append([j * Lx, (j + 2) * Lx - 1])         # across boundary (x)
            linkNNA_PBC.append([j * Lx, (j + 1) * Lx])              # across boundary (x)
            for i in range(1, Lx):
                linkNNA_PBC.append([j * Lx + i, (j + 1) * Lx + i - 1])
                linkNNA_PBC.append([j * Lx + i, (j + 1) * Lx + i])
        elif j % 2 == 1:
            for i in range(0, Lx - 1):
                linkNNA_PBC.append([j * Lx + i, (j + 1) * Lx + i])
                linkNNA_PBC.append([j * Lx + i, (j + 1) * Lx + i + 1])
            linkNNA_PBC.append([j * Lx + Lx - 1, (j + 1) * Lx + Lx - 1])    # across boundary (x)
            linkNNA_PBC.append([j * Lx + Lx - 1, (j + 1) * Lx])         # across boundary (x)

    # Across boundary (y)
    for i in range(Lx - 1):
        linkNNA_PBC.append([(Ly - 1) * Lx + i, i])
        linkNNA_PBC.append([(Ly - 1) * Lx + i, i + 1])
    linkNNA_PBC.append([Ly * Lx - 1, Lx - 1])       # across boundary (x)
    linkNNA_PBC.append([Ly * Lx - 1, 0])        # across boundary (x)
    return np.array(linkNNA_PBC)

def makeH_sp(Lx, Ly, Jx, Ja):
    nQ = int(Lx * Ly)
    H_dg = np.zeros([2 ** nQ])

    linkNNX = makeLinkNNX_PBC(Lx, Ly)
    linkNNA = makeLinkNNA_PBC(Lx, Ly)

    diagZ = np.array([1, -1])
    diagI = np.array([1, 1])

    for i in range(len(linkNNX)):
        _h_ = 1
        for j in range(nQ):
            if j in linkNNX[i]:
                _h_ = kron(diagZ, _h_)
            else:
                _h_ = kron(diagI, _h_)
        H_dg += Jx * _h_

    for i in range(len(linkNNA)):
        _h_ = 1
        for j in range(nQ):
            if j in linkNNA[i]:
                _h_ = kron(diagZ, _h_)
            else:
                _h_ = kron(diagI, _h_)
        H_dg += Ja * _h_

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