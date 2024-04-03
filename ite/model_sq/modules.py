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

def makeH(numQ, J, K, Kp, linkJ, linkK, linkKp):
    H = np.zeros((2 ** numQ, 1))

    for i in range(len(linkJ)):
        nnxZZ = 1
        for j in range(numQ):
            if j in linkJ[i]:
                nnxZZ = kron(diagZ, nnxZZ)  # Little endian order: A2 ※ A1 ※ A0 ===> nnxZZ on the right
            else:
                nnxZZ = kron(diagI, nnxZZ)
        H += J * nnxZZ

    for i in range(len(linkK)):
        nnaZZ = 1
        for j in range(numQ):
            if j in linkK[i]:
                nnaZZ = kron(diagZ, nnaZZ)
            else:
                nnaZZ = kron(diagI, nnaZZ)
        H += K * nnaZZ

    for i in range(len(linkKp)):
        nnaZZ = 1
        for j in range(numQ):
            if j in linkKp[i]:
                nnaZZ = kron(diagZ, nnaZZ)
            else:
                nnaZZ = kron(diagI, nnaZZ)
        H += Kp * nnaZZ

    e0 = np.min(H)
    e1 = 1e4
    for i in range(2 ** numQ):
        if H[i][0] < e1 and H[i][0] != e0:
            e1 = H[i][0]

    return H, e0, e1


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
