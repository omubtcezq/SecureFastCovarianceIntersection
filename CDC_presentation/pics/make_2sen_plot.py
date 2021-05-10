import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",   
    'font.family': 'serif',         # Use serif/main font for text elements
    'text.usetex': True,            # Use inline maths for ticks
    'pgf.rcfonts': False,           # Don't setup fonts from matplotlib rc params
})

fig = plt.figure()
fig.set_size_inches(w=4, h=3)
ax = fig.add_subplot(111)

w_quant = 0.1
w_steps = np.arange(0, 1+w_quant, w_quant)
print("w_steps:", w_steps)
print("1-w_steps:", 1-w_steps)

trA = 7.6
trB = 2.4

A = w_steps * trA
B = (1-w_steps) * trB

print("trA_w_options:", A)
print("trB_w_options:", B)

cmp_list = A+B
print(cmp_list)

w_vals = w_steps*trA > (1-w_steps)*trB
print(w_vals)
l = r = 0
for i,b in enumerate(w_vals):
    if not b:
        l = r = i
    else:
        r = i
        break
l = w_steps[l]
r = w_steps[r]


ax.plot(w_steps, A, marker='.', c='g', label=r'$L_1$', zorder=3)
ax.plot(w_steps, B, marker='.', c='b', label=r'$L_2 (\mathsf{reversed})$', zorder=3)
#ax.plot([trB/(trA+trB), trB/(trA+trB)],[0, trB/(trA+trB)*trA], linestyle='--', c='r')
#ax.scatter([trB/(trA+trB)],[trB/(trA+trB)*trA], marker='x', c='r', zorder=10)

ax.scatter([l,r],[0, 0], marker='x', c='grey', zorder=2, label=r'Solution limits')
ax.plot([l, l],[0, (1-l)*trB], linestyle='--', c='grey', zorder=2)
ax.plot([r, r],[0, r*trA], linestyle='--', c='grey', zorder=2)
ax.scatter([0.5*(l+r)],[0], marker='x', c='r', zorder=1, label=r'Approx. solution')

#plt.ylim(bottom=0)

plt.xlabel(r'$\omega_1$')

plt.legend(numpoints=1)

ax.yaxis.set_ticks([trB, trA])
ax.set_yticklabels([r'$\mathsf{tr}(\mathbf{P}_2)$', r'$\mathsf{tr}(\mathbf{P}_1)$'])

plt.tight_layout()

plt.savefig("2sen_weight_approx.pdf")
plt.close()