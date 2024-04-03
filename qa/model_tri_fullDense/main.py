# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ QA for the triangular model by exact dense evolution operator
#                   - This program costs huge time when the number of qubits > 8
#                   - Therefore it just serves as a verification
#                           for the main method
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Sep 6, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import model
import matplotlib.pyplot as plt
import tensorcircuit as tc
from time import time
from jax.scipy.linalg import expm

tc.set_backend("jax")
tc.set_dtype("complex128")

ds = 1e-4
dt = 0.1
N = int(1 / ds)

Jx, Ja = 0.9, 1.0
Lx, Ly = 3, 4
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
        # Calculate energy
        energy = calcEnergy(vec, H_sp)
        E_list.append(energy)

        # Update Hamiltonian and do evolution
        H = s * H0 + (1 - s) * H1
        evoOp = expm(-1j * H * dt)
        vec = evoOp @ vec

        # Move to next step
        print(f"\rs = {s}\t\tE = {energy}", end="")
        s += ds

    model.saveState(vec, Lx, Ly, ds, dt, Jx, Ja)
    print("")

    return E_list

if __name__ == "__main__":
    H_sp, H_dg, E0, E1 = model.makeH_sp(Lx, Ly, Jx, Ja)
    H0 = tc.backend.to_dense(H_sp)
    H1 = model.makeH_aux(nQ)

    # ------------------------------------------------------
    #   Evolutions and data saving
    # ------------------------------------------------------
    if reEvolve:
        print("Implementing exact QA...")
        print(f"\tQA, {Lx} × {Ly} triangular lattice\n\tJx = {Jx}, Ja = {Ja}, ds = {ds}, dt = {dt}")
        time_0 = time()

        model.saveE_list(f"fullQA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}", QA_tri())

        print("-" * 77)
        time_1 = time()
        model.reportTimeUsed(time_0, time_1)

    # ------------------------------------------------------
    #   Plotting and figure saving
    # ------------------------------------------------------
    if plot:
        E_list = model.loadE_list(f"fullQA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}")

        print("Exact energy:")
        print(f"\tE0 = {E0}\n\tE1 = {E1}")
        print(f"Final energies derived by QA: {E_list[-1]}")

        plt.figure(figsize=(6.5, 4.5), dpi=100)
        plt.subplots_adjust(left=0.12, bottom=0.13, right=0.93, top=0.9)

        plt.plot(
            [i * ds for i in range(N)], E_list,
            color="#B22222", linewidth=1.0, linestyle='-',
            marker="^", markersize=7, markevery=int(N/10),
            label="QA"
        )
        plt.plot(
            [i * ds for i in range(N)], [E0 for _ in range(N)],
            color="#0000FF", linewidth=1.0, linestyle='--', label=r"$E_0$"
        )
        plt.plot(
            [i * ds for i in range(N)], [E1 for _ in range(N)],
            color="#006400", linewidth=1.0, linestyle=':', label=r"$E_1$"
        )

        plt.xlabel(r"$s$", fontsize=13)
        plt.ylabel("Energy", fontsize=13)
        plt.title(f"Dense QA, {Lx} × {Ly}, Jx = {Jx}, Ja = {Ja}, ds = {ds}, dt = {dt}", fontsize=13)
        plt.legend(fontsize=13)
        plt.savefig(f"./figs/fullQA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}.eps", dpi=300)
        plt.show()