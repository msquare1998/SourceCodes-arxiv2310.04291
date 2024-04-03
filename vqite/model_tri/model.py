import jax.numpy as jnp
import numpy as np
from numpy import kron
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

def saveVec(vec, numQ):
    f0 = open('./finalVecReal.txt', 'w', encoding='utf-8')
    f1 = open('./finalVecImag.txt', 'w', encoding='utf-8')
    for i in range(2 ** numQ):
        f0.write(str(vec[i][0].real) + '\n')
        f1.write(str(vec[i][0].imag) + '\n')
    f0.close()
    f1.close()

def normalization(vec):
    nVec = np.zeros(len(vec)).astype(complex)
    for i in range(len(vec)):
        nVec[i] = vec[i] / tc.backend.norm(vec)
    nVec = tc.array_to_tensor(nVec)
    nVec = tc.backend.reshape(nVec, [-1])
    return nVec

def checkNormalized(vec):
    if abs(1 - tc.backend.norm(vec)) > 1e-5:
        raise Exception("\nFail to normalize the state; Factor = {}".format(tc.backend.norm(vec)))

def makeIniVec(numQ):
    superState = 1 / 2 ** 0.5 * jnp.ones(2)
    psi0 = 1
    for _ in range(numQ):
        psi0 = kron(superState, psi0)
    return tc.array_to_tensor(psi0)

def makeIniVec0(numQ):
    state = np.ones(2)
    state[1] = 0
    psi0 = 1
    for _ in range(numQ):
        psi0 = kron(state, psi0)
    return tc.array_to_tensor(psi0)

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
    linkNNX = makeLinkNNX_PBC(Lx, Ly)
    linkNNA = makeLinkNNA_PBC(Lx, Ly)

    nQ = int(Lx * Ly)

    H_dg = np.zeros([2 ** nQ])
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

# -------------------------------------------------------------------------------------------------------
#   Data saving, loading, and Timer
# -------------------------------------------------------------------------------------------------------
def saveState(vec, nQ, Lx, Ly, Jx, Ja, dTau, stdDev):
    f_r = open(f'./stateVecInfo/realPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}_stdDev{stdDev}.dat', 'w', encoding='utf-8')
    f_i = open(f'./stateVecInfo/imagPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}_stdDev{stdDev}.dat', 'w', encoding='utf-8')
    for i in range(2 ** nQ):
        f_r.write(str(vec[i].real) + '\n')
        f_i.write(str(vec[i].imag) + '\n')
    f_r.close()
    f_i.close()

def loadState(nQ, Lx, Ly, Jx, Ja, dTau, stdDev):
    f_r = open(f'./stateVecInfo/realPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}_stdDev{stdDev}.dat', 'r', encoding='utf-8')
    f_i = open(f'./stateVecInfo/imagPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}_stdDev{stdDev}.dat', 'r', encoding='utf-8')
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