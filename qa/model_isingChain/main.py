from jax import numpy as jnp
import matplotlib.pyplot as plt
import tensorcircuit as tc
from time import time
import model

tc.set_backend("jax")
tc.set_dtype("complex128")

nQ = 16
ds, dt = 1e-4, 0.1
N = int(1 / ds)
J = 1.0
reEvolve, plot = False, True

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
        lineWidth = 2.5
        E_list_trotterized_circuit = model.loadE_lsit(f"trotterized_circuit_nQ{nQ}_ds{ds}_dt{dt}")

        print("Exact energy:")
        print(f"\tE0 = {E0}\n\tE1 = {E1}")
        print("Final energies derived by the three methods:")
        print(f"\tQA (trotterized circuit): {E_list_trotterized_circuit[-1]}")

        fig = plt.figure(figsize=(7.5, 5.0), dpi=300)
        plotParams = {
            'axes.labelsize': 20,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 18,
        }
        plt.rcParams.update(plotParams)
        #ax1 = plt.subplots_adjust(left=0.2, bottom=0.13, right=0.93, top=0.9)
        ax1 = plt.subplots_adjust(left=0.16, bottom=0.13, right=0.96, top=0.99)

        plt.plot(
            [i * dt for i in range(N)], E_list_trotterized_circuit,
            color="#B22222", linewidth=lineWidth, linestyle='-',
            #marker="s", markersize=10, markevery=int(N / 10 - 1),
            label=r"QA"
        )
        plt.plot(
            [i * dt for i in range(N)], [E0 for _ in range(N)],
            color="black", linewidth=lineWidth, linestyle='-.', label=r"$E_0$"
        )
        """plt.plot(
            [i * ds for i in range(N)], [E1 for _ in range(N)],
            color="#1E90FF", linewidth=1.0, linestyle=':', label=r"$E_1$"
        )"""

        plt.xlabel(r"$T$")
        plt.ylabel(r"$\langle H_0\rangle$")
        #plt.title(
        #    f"{nQ}-qubit Ising chain (PBC)",
        #    fontsize=13
        #)

        plt.legend()
        plt.savefig("./figs/QA_isingChain_nQ{}_ds{}_dt{}.pdf".format(nQ, ds, dt), dpi=300)
        plt.show()