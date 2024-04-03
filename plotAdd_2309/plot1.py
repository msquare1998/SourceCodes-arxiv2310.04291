# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ Some additional plots about the paper
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Sep 18, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import matplotlib.pyplot as plt
from header import loadList, getE0E1

""" ------------------------------------------------------------------------
----------------------------------------------------------------------------
        @ Public params
----------------------------------------------------------------------------
------------------------------------------------------------------------ """
Lx, Ly = 4, 4
Jx, Ja = 0.9, 1.0
E0, E1 = getE0E1(Lx, Ly, Jx, Ja)

""" ------------------------------------------------------------------------
----------------------------------------------------------------------------
        @ QA params
----------------------------------------------------------------------------
------------------------------------------------------------------------ """
dt = 0.1
ds = 0.0025
N = int(1 / ds)
print(f"T for QA: {int(1 / ds * dt)}")

""" ------------------------------------------------------------------------
----------------------------------------------------------------------------
        @ VQITE params
----------------------------------------------------------------------------
------------------------------------------------------------------------ """
stepsCut = 400
dTau = 0.1
stdDev = 0.05

lineWidth = 2.5

E_list_ITE = loadList(f'../ite/model_tri/exactEnergyList/ITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}.dat')
E_list_VQITE = loadList(f'../vqite/model_tri/energyList/VQITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}_stdDev{stdDev}.dat')
E_list_QA = loadList(f'../qa/model_tri/energyList/QA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}.dat')
E_list_DIAG_VQITE = loadList(f'../diag-vqite/model_tri/energyList/VQITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau0.05_stdDev{stdDev}.dat')

print("Final energy:")
print(f"\tITE: {E_list_ITE[-1]}\n\tVQITE: {E_list_VQITE[-1]}\n\tQA: {E_list_QA[-1]}")

fig = plt.figure(figsize=(7.5, 5.0), dpi=300)
plotParams = {
    'axes.labelsize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 18,
}
plt.rcParams.update(plotParams)

# ------------------------------------------------------------------
#       ax1
# ------------------------------------------------------------------
ax1 = plt.subplots_adjust(left=0.16, bottom=0.13, right=0.96, top=0.99)

#titleName = "(a)"
#plt.title(f"{titleName: <60}", fontsize=25)

plt.plot(
    [i * dt for i in range(N)], E_list_QA,
    color="#B22222", linewidth=lineWidth, linestyle='-',
    #marker="s", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"QA"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_ITE[:stepsCut],
    color="#1E90FF", linewidth=lineWidth, linestyle='-',
    #marker="o", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"QITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_VQITE[:stepsCut],
    color="#EEAD0E", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"VQITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_DIAG_VQITE[:stepsCut],
    color="#339933", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"Diag-VQITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], [E0 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle='-.', label=r'$E_0$'
)

plt.plot(
    [i * dTau for i in range(stepsCut)], [E1 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle=':', label=r'$E_1$'
)

plt.xlabel(r"$T$")
plt.ylabel(r"$\langle \hat H_0\rangle$")
plt.legend()
# ------------------------------------------------------------------
#       ax2
# ------------------------------------------------------------------
left, bottom, width, height = 0.41, 0.78, 0.19, 0.15
ax2 = fig.add_axes([left, bottom, width, height])
plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_ITE[:stepsCut],
    color="#1E90FF", linewidth=lineWidth, linestyle='-',
    #marker="o", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"QITE"
)

ax2.plot(
    [i * dTau for i in range(stepsCut)], E_list_VQITE[:stepsCut],
    color="#EEAD0E", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"VQITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_DIAG_VQITE[:stepsCut],
    color="#339933", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"Diag-VQITE"
)

ax2.plot(
    [i * dTau for i in range(stepsCut)], [E0 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle='-.', label=r'$E_0$'
)

ax2.plot(
    [i * dTau for i in range(stepsCut)], [E1 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle=':', label=r'$E_1$'
)

plt.xlim((0, 5))
plt.ylim((-18, -13))


plt.savefig(f"./figs/QA_ITE_VQITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_ds{ds}_dt{dt}.pdf", dpi=300)
plt.show()