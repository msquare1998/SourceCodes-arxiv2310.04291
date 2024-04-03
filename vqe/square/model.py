import numpy as np
from numpy import kron
import tensorcircuit as tc

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

plaquette = np.array([
    # For [a, b, c, d], {a, b} are in linkK, {c, d} are in linkJ
    [0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [3, 0, 7, 4],
    [8, 9, 4, 5], [9, 10, 5, 6], [10, 11, 6, 7], [11, 8, 7, 4],
    [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15], [11, 8, 15, 12],
    [0, 1, 12, 13], [1, 2, 13, 14], [2, 3, 14, 15], [3, 0, 15, 12]
])


"""def makelink2DSquare(Lx, Ly):
    bSites = []
    # Horizontal
    for j in range(Ly):
        for i in range(Lx):
            bSites.append([Lx * j + i, Lx * j + (i + 1) % Lx])

    # Vertical
    for j in range(Ly - 1):
        for i in range(Lx):
            bSites.append([Lx * j + i, Lx * j + Lx + i])
    for i in range(Lx):
        bSites.append([(Ly - 1) * Lx + i, i])
    return np.array(bSites)"""

def makeIniVec(nQ):
    plusState = 1 / 2 ** 0.5 * np.ones(2)
    psi0 = 1
    for _ in range(nQ):
        psi0 = kron(plusState, psi0)
    return psi0

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

    tc.set_backend("tensorflow")
    tc.set_dtype("complex128")
    h = tc.backend.coo_sparse_matrix(indices=H_sp_indices, values=H_dg, shape=[2 ** nQ, 2 ** nQ])

    e0 = np.min(H_dg)
    e1 = 1e4
    for i in range(2 ** nQ):
        if H_dg[i] < e1 and H_dg[i] != e0:
            e1 = H_dg[i]
    return h, H_dg, e0, e1



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