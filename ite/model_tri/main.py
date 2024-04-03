import numpy as np
import modules as md
from time import time

Jx, Ja = 0.9, 1.0
Lx, Ly = 4, 4
dTau, steps = 0.05, 1000
nQ = int(Lx * Ly)

reEvolve, count = False, True

def calcEnergy(numQ, vec, H):
    dim = 2 ** numQ
    E = 0
    for i in range(dim):
        E += H[i][0] * (abs(vec[i][0]) ** 2)
    return E

def main():
    linkNNX = md.makeLinkNNX_PBC(Lx, Ly)
    linkNNA = md.makeLinkNNA_PBC(Lx, Ly)

    stateVec = md.makeIniVec(nQ)
    H = md.makeH(nQ, Jx, Ja, linkNNX, linkNNA)

    imgEvo = md.makeImEvoOperator(nQ, dTau, H)

    f0 = open(f"./exactEnergyList/ITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}.dat", "w", encoding="utf-8")

    for i in range(steps):
        print("τ = {}...".format(i * dTau), end="\t")
        energy = calcEnergy(nQ, stateVec, H)
        print("Exact energy = {}".format(energy))
        f0.write(str(energy) + "\n")

        stateVec = stateVec * imgEvo
        md.normalization(nQ, stateVec)
        md.checkNormalized(nQ, stateVec)

    f0.close()
    print("* The list of energies has been saved.")
    md.saveState(stateVec, nQ, Lx, Ly, Jx, Ja)
    print("* The final state has been saved.")

if __name__ == "__main__":
    startTime = time()
    main()
    endTime = time()
    md.reportTimeUsed(startTime, endTime)
