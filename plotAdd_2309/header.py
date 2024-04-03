import numpy as np
from numpy import kron
import tensorcircuit as tc

# ----------------------------------------------------------------------
#       Get E0, E1 for the triangular model
# ----------------------------------------------------------------------
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

def getE0E1(Lx, Ly, Jx, Ja):
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
    return e0, e1

# ----------------------------------------------------------------------
#       Get E0, E1 for the square model
# ----------------------------------------------------------------------
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

def getE0E1_sq(Lx, Ly, J, K, Kp):
    # ◆◆◆◆◆◆◆◆◆◆◆◆◆ modified, compared to "2D_3params" ◆◆◆◆◆◆◆◆◆◆◆◆◆
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

    tc.set_backend("jax")
    tc.set_dtype("complex128")
    h = tc.backend.coo_sparse_matrix(indices=H_sp_indices, values=H_dg, shape=[2 ** nQ, 2 ** nQ])

    e0 = np.min(H_dg)
    e1 = 1e4
    for i in range(2 ** nQ):
        if H_dg[i] < e1 and H_dg[i] != e0:
            e1 = H_dg[i]
    return e0, e1



# ----------------------------------------------------------------------
#       Save and load data
# ----------------------------------------------------------------------
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
