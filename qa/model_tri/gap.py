from model import makeLinkNNX_PBC, makeLinkNNA_PBC
import numpy as np
from numpy import kron
def calcEnergies(Lx, Ly, Jx, Ja):
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

    e0 = np.min(H_dg)
    e1 = 1e4
    for i in range(2 ** nQ):
        if H_dg[i] < e1 and H_dg[i] != e0:
            e1 = H_dg[i]
    return e0, e1


Jx = 0.9
Ja = 1.0

Lx, Ly = 5, 6
E0, E1 = calcEnergies(3, 4, Jx, Ja)
print(f"{Lx} * {Ly}:\nE1 = {E1}, E0 = {E0}, gap = {E1 - E0}")

Lx, Ly = 4, 4
E0, E1 = calcEnergies(4, 4, Jx, Ja)
print(f"{Lx} * {Ly}:\nE1 = {E1}, E0 = {E0}, gap = {E1 - E0}")