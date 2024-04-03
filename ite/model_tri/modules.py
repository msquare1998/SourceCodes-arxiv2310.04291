import numpy as np
from numpy import kron, array
from math import exp


''' ---------------------------------------------------------
* Basic
--------------------------------------------------------- '''
def sqrt(x):
    return x ** 0.5


diagZ = array([[1], [-1]])
diagI = array([[1], [1]])


superState = array([[1 / sqrt(2)], [1 / sqrt(2)]])
downState = array([[0], [1]])


''' ---------------------------------------------------------
* Printing
--------------------------------------------------------- '''
def printLinkNNX_PBC(Lx, linkNNX_PBC):
    for i in range(len(linkNNX_PBC)):
        print(linkNNX_PBC[i], end="")
        if (i + 1) % (Lx) == 0:
            print("", end="\n")
        else:
            print("", end=", ")


def printLinkNNA_PBC(Lx, linkNNA_PBC):
    for i in range(len(linkNNA_PBC)):
        print(linkNNA_PBC[i], end="")
        if (i + 1) % (2 * Lx) == 0:
            print("", end="\n")
        else:
            print("", end=", ")


''' ---------------------------------------------------------
* Initialization
--------------------------------------------------------- '''
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

def makeIniVec(numQ):
    vec = 1
    for _ in range(numQ):
        vec = kron(superState, vec)
    return vec


def makeImEvoOperator(numQ, tau, H):
    evo = np.zeros((2 ** numQ, 1))
    for i in range(2 ** numQ):
        evo[i][0] = exp(-tau * H[i][0])
    return evo


def makeH(numQ, Jx, Ja, linkNNX, linkNNA):
    # diagonal ==> 1D array
    H = np.zeros((2 ** numQ, 1))

    for i in range(len(linkNNX)):
        h = 1
        for j in range(numQ):
            if j in linkNNX[i]:
                h = kron(diagZ, h)  # Little endian order: A2 ※ A1 ※ A0 ===> nnxZZ on the right
            else:
                h = kron(diagI, h)
        H += Jx * h

    for i in range(len(linkNNA)):
        h = 1
        for j in range(numQ):
            if j in linkNNA[i]:
                h = kron(diagZ, h)
            else:
                h = kron(diagI, h)
        H += Ja * h

    return H


''' ---------------------------------------------------------
* Aux functions
--------------------------------------------------------- '''
def checkArgs(Lx, Ly, dTau):
    if (Lx < 2) or (Ly < 2):
        raise Exception("Lx and Ly must be integers that > 1.")
    if (Ly % 2) != 0:
        raise Exception("Ly must be an even integer.")
    if dTau < 1:
        raise Exception("dTau must be at least 1.")


def checkNormalized(numQ, vec):
    sum = 0
    for i in range(2 ** numQ):
        sum += (abs(vec[i])) ** 2
    if abs(sum - 1) > 10 ** (-9):
        raise Exception("Normalization Error.")


def normalization(numQ, vec):
    factor = 0
    for i in range(2 ** numQ):
        factor += (abs(vec[i][0]) ** 2)
    for j in range(2 ** numQ):
        vec[j][0] /= sqrt(factor)
    return vec


def countPreserved(numQ, vec):
    e = 10 ** (-9)
    count = 0
    for i in range(2 ** numQ):
        if abs(vec[i][0]) > e:
            count += 1
    return count


def reportTimeUsed(t0, t1):
    dt = t1 - t0
    h = int(dt / 3600)
    m = int((dt - h * 3600) / 60)
    s = dt - 3600 * h - 60 * m
    print("Time used: %dh, %dm, %.2fs" % (h, m, s))

def saveState(vec, nQ, Lx, Ly, Jx, Ja):
    f_r = open(f'./stateVecInfo/realPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}.dat', 'w', encoding='utf-8')
    f_i = open(f'./stateVecInfo/imagPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}.dat', 'w', encoding='utf-8')
    for i in range(2 ** nQ):
        f_r.write(str(vec[i].real) + '\n')
        f_i.write(str(vec[i].imag) + '\n')
    f_r.close()
    f_i.close()

def loadState(nQ, Lx, Ly, Jx, Ja):
    f_r = open(f'./stateVecInfo/realPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}.dat', 'r', encoding='utf-8')
    f_i = open(f'./stateVecInfo/imagPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}.dat', 'r', encoding='utf-8')
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

def saveGS(vec_real, nQ, Lx, Ly, Jx, Ja, label):
    f_r = open(f'./GS_tri_info/GS_{label}_realPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}.dat', 'w', encoding='utf-8')
    f_i = open(f'./GS_tri_info/GS_{label}_imagPart_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}.dat', 'w', encoding='utf-8')
    for i in range(2 ** nQ):
        f_r.write(str(vec_real[i]) + '\n')
        f_i.write(str(0) + '\n')
    f_r.close()
    f_i.close()
    print("The exact ground state has been saved.")