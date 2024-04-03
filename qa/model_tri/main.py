# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ QA for the triangular model
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Aug 15, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import model
import matplotlib.pyplot as plt
import tensorcircuit as tc
from time import time

tc.set_backend("jax")
tc.set_dtype("complex128")

ds = 1e-5
dt = 0.1
N = int(1 / ds)

Jx, Ja = 0.9, 1.0
Lx, Ly = 4, 4
nQ = int(Lx * Ly)

reEvolve, plot = False, True

@tc.backend.jit
def calcEnergy(vec, h_sp):
    return (tc.backend.conj(tc.backend.transpose(vec)) @ h_sp @ vec).real

def QA_tri():
    vec = model.makeIniVec(nQ)

    s = 0
    E_list = []

    for i in range(N):
        # ---------------------------------
        # expm(-i H_1 dt / 2)
        # ---------------------------------
        circ = tc.Circuit(nQ, inputs=vec)
        for q in range(nQ):
            circ.rx(q, theta=2 * (1 - s) * dt / 2)
        vec = circ.state()

        # ---------------------------------
        # expm(-i H_0 dt)
        # ---------------------------------
        vec = model.makeDiagEvoOp(nQ, s * H_dg, dt) * vec

        # ---------------------------------
        # expm(-i H_1 dt / 2)
        # ---------------------------------
        circ = tc.Circuit(nQ, inputs=vec)
        for q in range(nQ):
            circ.rx(q, theta=2 * (1 - s) * dt / 2)

        # ---------------------------------
        # Calculating energy
        # ---------------------------------
        energy = calcEnergy(circ.state(), H_sp)
        E_list.append(energy)

        print(f"\rs = {s}\t\tE = {energy}", end="\t")
        vec = circ.state()

        s += ds

    model.saveState(vec, Lx, Ly, ds, dt, Jx, Ja)
    print("")

    return E_list

if __name__ == "__main__":
    H_sp, H_dg, E0, E1 = model.makeH_sp(Lx, Ly, Jx, Ja)

    # ------------------------------------------------------
    #   Evolutions and data saving
    # ------------------------------------------------------
    if reEvolve:
        print("Implementing QA under a trotterized Hamiltonian and quantum circuit...")
        print(f"\tQA, {Lx} × {Ly} triangular lattice\n\tJx = {Jx}, Ja = {Ja}, ds = {ds}, dt = {dt}")
        time_0 = time()

        model.saveE_list(f"QA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}", QA_tri())

        print("-" * 77)
        time_1 = time()
        model.reportTimeUsed(time_0, time_1)

    # ------------------------------------------------------
    #   Plotting and figure saving
    # ------------------------------------------------------
    if plot:
        lineWidth = 2.5

        E_list = model.loadE_list(f"QA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}")

        print("Exact energy:")
        print(f"\tE0 = {E0}\n\tE1 = {E1}")
        print(f"Final energies derived by QA: {E_list[-1]}")

        fig = plt.figure(figsize=(7.5, 5.0), dpi=300)
        plotParams = {
            'axes.labelsize': 20,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 18,
        }
        plt.rcParams.update(plotParams)
        ax1 = plt.subplots_adjust(left=0.16, bottom=0.13, right=0.96, top=0.99)

        plt.plot(
            [i * dt for i in range(N)], E_list,
            color="#B22222", linewidth=lineWidth, linestyle='-',
            #marker="s", markersize=10, markevery=int(N/10 - 1),
            label="QA"
        )
        plt.plot(
            [i * dt for i in range(N)], [E0 for _ in range(N)],
            color="black", linewidth=lineWidth, linestyle='-.', label=r"$E_0$"
        )
        plt.plot(
            [i * dt for i in range(N)], [E1 for _ in range(N)],
            color="black", linewidth=lineWidth, linestyle=':', label=r"$E_1$"
        )

        plt.xlabel(r"$T$")
        plt.ylabel(r"$\langle \hat H_0\rangle$")

        #plt.title(f"QA, {Lx} × {Ly}, Jx = {Jx}, Ja = {Ja}, ds = {ds}, dt = {dt}", fontsize=13)
        plt.legend()
        plt.savefig(f"./figs/QA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}.eps", dpi=300)
        plt.show()
