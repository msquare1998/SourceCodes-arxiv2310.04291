# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ Three different ways to
#               implement Quantum Anealing (QA) for an 1D Ising chain (PBC)
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Aug 15, 2023
# ███████████████████████████████████████████████████████████████████████████████████
from jax import numpy as jnp
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt
import tensorcircuit as tc
from time import time
import model

tc.set_backend("jax")
tc.set_dtype("complex128")

nQ = 8
ds, dt = 1e-4, 0.1
N = int(1 / ds)
J = -1.0
reEvolve, plot = False, True

def QA_exact_expm():
    H0 = jnp.array(tc.backend.to_dense(H_sp))
    H_aux = jnp.array(model.makeH_aux(nQ))
    vec = jnp.array(model.makeIniVec_old(nQ))
    s = 0

    E_list = []

    for i in range(N):
        energy = model.calcEnergy_old(H0, vec)
        H = s * H0 + (1 - s) * H_aux
        print(f"\rs = {s}\t\tE = {energy}", end="")
        E_list.append(energy)

        evo = expm(-1j * H * dt)
        vec = evo @ vec
        s += ds

    print("")
    return E_list
def QA_trotterized_expm():
    H0 = jnp.array(tc.backend.to_dense(H_sp))
    H_aux = jnp.array(model.makeH_aux(nQ))
    vec = jnp.array(model.makeIniVec_old(nQ))
    s = 0

    E_list = []

    for i in range(N):
        energy = model.calcEnergy_old(H0, vec)
        print(f"\rs = {s}\t\tE = {energy}", end="")
        E_list.append(energy)

        evoZ = expm(-1j * s * H0 * dt)
        evoX = expm(-1j * (1 - s) * H_aux * dt / 2)
        vec = evoX @ evoZ @ evoX @ vec

        s += ds
    print("")
    return E_list

@tc.backend.jit
def calcEnergy(vec, h_sp):
    return (tc.backend.conj(tc.backend.transpose(vec)) @ h_sp @ vec).real

def QA_trotterized_circuit():
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
        print(f"\rs = {s}\t\tE = {energy}", end="")
        vec = circ.state()

        s += ds
    print("")

    return E_list

if __name__ == "__main__":
    H_sp, H_dg, E0, E1 = model.makeH_1D_Ising_PBC(nQ, J)
    # ------------------------------------------------------
    #   Evolutions and data saving
    # ------------------------------------------------------
    if reEvolve:
        links_PBC = jnp.array([[i, (i + 1) % nQ] for i in range(nQ)])

        print("-" * 77)
        print("Implementing QA under an exact Hamiltonian and matrix exponential...")
        time_0 = time()
        model.saveE_list(f"exact_expm_nQ{nQ}_ds{ds}_dt{dt}", QA_exact_expm())
        time_1 = time()
        model.reportTimeUsed(time_0, time_1)

        print("-" * 77)
        print("Implementing QA under a trotterized Hamiltonian and matrix exponential...")
        time_0 = time()
        model.saveE_list(f"trotterized_expm_nQ{nQ}_ds{ds}_dt{dt}", QA_trotterized_expm())
        time_1 = time()
        model.reportTimeUsed(time_0, time_1)

        print("-" * 77)
        print("Implementing QA under a trotterized Hamiltonian and quantum circuit...")
        time_0 = time()
        model.saveE_list(f"trotterized_circuit_nQ{nQ}_ds{ds}_dt{dt}", QA_trotterized_circuit())
        time_1 = time()
        model.reportTimeUsed(time_0, time_1)

        print("-" * 77)

    # ------------------------------------------------------
    #   Plotting and figure saving
    # ------------------------------------------------------
    if plot:
        E_list_exact_expm = model.loadE_lsit(f"exact_expm_nQ{nQ}_ds{ds}_dt{dt}")
        E_list_trotterized_expm = model.loadE_lsit(f"trotterized_expm_nQ{nQ}_ds{ds}_dt{dt}")
        E_list_trotterized_circuit = model.loadE_lsit(f"trotterized_circuit_nQ{nQ}_ds{ds}_dt{dt}")

        print("Exact energy:")
        print(f"\tE0 = {E0}\n\tE1 = {E1}")
        print("Final energies derived by the three methods:")
        print(f"\tQA (exact expm): {E_list_exact_expm[-1]}")
        print(f"\tQA (trotterized expm): {E_list_trotterized_expm[-1]}")
        print(f"\tQA (trotterized circuit): {E_list_trotterized_circuit[-1]}")

        plt.figure(figsize=(6.5, 4.5), dpi=100)
        plt.subplots_adjust(left=0.12, bottom=0.13, right=0.93, top=0.9)
        plt.plot(
            [i * ds for i in range(N)], E_list_exact_expm,
            color="#FFB90F", linewidth=1.0, linestyle='-',
            marker="s", markersize=7, markevery=500,
            label="QA"
        )

        plt.plot(
            [i * ds for i in range(N)], [E0 for _ in range(N)],
            color="#0000FF", linewidth=1.0, linestyle='--', label=r"$E_0$"
        )

        plt.xlabel(r"$s$", fontsize=13)
        plt.ylabel("Energy", fontsize=13)
        plt.title(
            f"{nQ}-qubit Ising chain (PBC)",
            fontsize=13
        )

        plt.legend(fontsize=13)
        plt.savefig("./figs/paramsTest_{}_ds{}_dt{}.eps".format(nQ, ds, dt), dpi=300)
        plt.show()