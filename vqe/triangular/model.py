import numpy as np
from numpy import kron
import tensorcircuit as tc

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

def makeIniVec(nQ):
    plusState = 1 / 2 ** 0.5 * np.ones(2)
    psi0 = 1
    for _ in range(nQ):
        psi0 = kron(plusState, psi0)
    return psi0

def makeH_sp(Lx, Ly, J, K, linkJ, linkK):
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

    H_sp_indices = np.array([[i, i] for i in range(2 ** nQ)])

    tc.set_backend("tensorflow")
    tc.set_dtype("complex128")
    h = tc.backend.coo_sparse_matrix(indices=H_sp_indices, values=H_dg, shape=[2 ** nQ, 2 ** nQ])

    e0 = np.min(H_dg)
    e1 = 1e4
    for i in range(2 ** nQ):
        if H_dg[i] < e1 and H_dg[i] != e0:
            e1 = H_dg[i]
    return h, e0, e1

def saveList(filePath, myList):
    f0 = open(f'{filePath}', 'w', encoding='utf-8')
    for e in myList:
        f0.write(str(e) + '\n')
    f0.close()

def loadList(filePath):
    f0 = open(f'{filePath}', 'r', encoding='utf-8')
    myList = []
    idx = 0
    while True:
        line0 = f0.readline()
        if line0:
            myList.append(float(line0.strip('\n')))
            idx += 1
        else:
            break
    f0.close()
    return myList

def reportTimeUsed(t0, t1):
    dt = t1 - t0
    h = int(dt / 3600)
    m = int((dt - h * 3600) / 60)
    s = dt - 3600 * h - 60 * m
    print("* Time used: %dh, %dm, %.2fs" % (h, m, s))
