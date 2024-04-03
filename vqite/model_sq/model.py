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

    tc.set_backend("jax")
    tc.set_dtype("complex128")
    h = tc.backend.coo_sparse_matrix(indices=H_sp_indices, values=H_dg, shape=[2 ** nQ, 2 ** nQ])

    e0 = np.min(H_dg)
    e1 = 1e4
    for i in range(2 ** nQ):
        if H_dg[i] < e1 and H_dg[i] != e0:
            e1 = H_dg[i]
    return h, H_dg, e0, e1


# -------------------------------------------------------------------------------------------------------
#   Data saving, loading, and Timer
# -------------------------------------------------------------------------------------------------------
def saveState(vec, nQ, J, K, Kp, dTau, stdDev):
    f_r = open(f'./stateVecInfo/realPart_J{J}_K{K}_K{Kp}_dTau{dTau}_stdDev{stdDev}.dat', 'w', encoding='utf-8')
    f_i = open(f'./stateVecInfo/imagPart_J{J}_K{K}_K{Kp}_dTau{dTau}_stdDev{stdDev}.dat', 'w', encoding='utf-8')
    for i in range(2 ** nQ):
        f_r.write(str(vec[i].real) + '\n')
        f_i.write(str(vec[i].imag) + '\n')
    f_r.close()
    f_i.close()

def loadState(nQ, J, K, Kp, dTau, stdDev):
    f_r = open(f'./stateVecInfo/realPart_J{J}_K{K}_K{Kp}_dTau{dTau}_stdDev{stdDev}.dat', 'r', encoding='utf-8')
    f_i = open(f'./stateVecInfo/imagPart_J{J}_K{K}_K{Kp}_dTau{dTau}_stdDev{stdDev}.dat', 'r', encoding='utf-8')
    vec = np.zeros([2 ** nQ]).astype(complex)
    idx = 0
    while True:
        line0 = f_r.readline()
        line1 = f_i.readline()
        if line0 and line1:
            vec[idx] = float(line0.strip('\n')) + 1j * float(line1.strip('\n'))
            idx += 1
        else:
            break
    f_i.close()
    f_r.close()
    return vec

def saveE_list(fileName, E_list):
    f0 = open(f'./energyList/{fileName}.dat', 'w', encoding='utf-8')
    for e in E_list:
        f0.write(str(e) + '\n')
    f0.close()

def loadE_list(fileName):
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

def load_otherList(fileName):
    f0 = open(f'{fileName}.dat', 'r', encoding='utf-8')
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

def countPreserved(vec, threshold):
    cc = 0
    print("State(s) preserved:")
    for i in range(len(vec)):
        prob = abs(vec[i]) ** 2
        if prob > threshold:
            print(f"\t|{bin(i)[2:]}〉with prob = {prob}")
            cc += 1
    return cc


def reportTimeUsed(t0, t1):
    dt = t1 - t0
    h = int(dt / 3600)
    m = int((dt - h * 3600) / 60)
    s = dt - 3600 * h - 60 * m
    print("Time used: %dh, %dm, %.2fs" % (h, m, s))