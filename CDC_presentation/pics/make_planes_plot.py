import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import numpy as np

SAVE_NOT_SHOW = True

if SAVE_NOT_SHOW:
	mpl.use("pgf")

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",   
    'font.family': 'serif',         # Use serif/main font for text elements
    'text.usetex': True,            # Use inline maths for ticks
    'pgf.rcfonts': False,           # Don't setup fonts from matplotlib rc params
})


fig = plt.figure()
fig.set_size_inches(w=12, h=3)


#========================================================================================#


ax = fig.add_subplot(131, projection='3d')

# fig.suptitle(r'First partial solution')
# ax.set_title(r'First Partial solution')

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)

ax.set_xlabel(r'$\omega_1$', rotation=0)
ax.set_ylabel(r'$\omega_2$', rotation=0)
ax.set_zlabel(r'$\omega_3$', rotation=0)
ax.view_init(elev=35, azim=25)
ax.dist = 12

# Solution plane
xy = np.array([[1,0],
               [0,0],
               [0,1]])
z = np.array([0,1,0])
trigs = [[0,1,2]]
triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
ax.plot_trisurf(triangles, z, color=(0.7,0.2,0.2,0.5), zorder=1)


# Partial solution 1
sol1 = 0.24
sol1Line, = ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='g')
ax.scatter([sol1, 0],[1-sol1, 0],[0, 1], c='g', marker='x', depthshade=False)

solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c=(0.7,0.2,0.2), marker = 'o')
# l = ax.legend([solutionSurfaceFakeLine, sol1Line], 
# 	[r'$\omega_1,\omega_2,\omega_3$ solution space', r'$\omega_1$, $\omega_2$ partial solution'], numpoints=1, loc=1, framealpha=0.6)

ax.xaxis.set_ticks([0,0.5,1])
ax.yaxis.set_ticks([0,0.5,1])
ax.zaxis.set_ticks([0,0.5,1])




#========================================================================================#




ax = fig.add_subplot(132, projection='3d')

#fig.suptitle(r'All partial solutions')
#ax.set_title(r'All partial solutions')

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)

ax.set_xlabel(r'$\omega_1$', rotation=0)
ax.set_ylabel(r'$\omega_2$', rotation=0)
ax.set_zlabel(r'$\omega_3$', rotation=0)
ax.view_init(elev=35, azim=25)
ax.dist = 12

# Solution plane
xy = np.array([[1,0],
               [0,0],
               [0,1]])
z = np.array([0,1,0])
trigs = [[0,1,2]]
triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
ax.plot_trisurf(triangles, z, color=(0.7,0.2,0.2,0.5), zorder=1)

# Partial solution 1
sol1 = 0.24
sol1Line, = ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='g')
ax.scatter([sol1, 0],[1-sol1, 0],[0, 1], c='g', marker='x', depthshade=False)

# Partial solution 2
sol2 = 0.52
sol2Line, = ax.plot([0, 1],[sol2, 0],[1-sol2, 0], linestyle='--', c='b')
ax.scatter([0, 1],[sol2, 0],[1-sol2, 0], c='b', marker='x', depthshade=False)

solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c=(0.7,0.2,0.2), marker = 'o')
# l = ax.legend([solutionSurfaceFakeLine, sol1Line, sol2Line], 
#           [r'$\omega_1,\omega_2,\omega_3$ solution space', r'$\omega_1$, $\omega_2$ partial solution', r'$\omega_2$, $\omega_3$ partial solution'], 
#           numpoints=1, framealpha=0.6)

ax.xaxis.set_ticks([0,0.5,1])
ax.yaxis.set_ticks([0,0.5,1])
ax.zaxis.set_ticks([0,0.5,1])




#========================================================================================#



ax = fig.add_subplot(133, projection='3d')

#fig.suptitle(r'All partial solutions')
#ax.set_title(r'All partial solutions')

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)

ax.set_xlabel(r'$\omega_1$', rotation=0)
ax.set_ylabel(r'$\omega_2$', rotation=0)
ax.set_zlabel(r'$\omega_3$', rotation=0)
ax.view_init(elev=35, azim=25)
ax.dist = 12

# Solution plane
xy = np.array([[1,0],
               [0,0],
               [0,1]])
z = np.array([0,1,0])
trigs = [[0,1,2]]
triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
ax.plot_trisurf(triangles, z, color=(0.7,0.2,0.2,0.5), zorder=1)

# Partial solution 1
sol1 = 0.24
sol1Line, = ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='g')
ax.scatter([sol1, 0],[1-sol1, 0],[0, 1], c='g', marker='x', depthshade=False)

# Partial solution 2
sol2 = 0.52
sol2Line, = ax.plot([0, 1],[sol2, 0],[1-sol2, 0], linestyle='--', c='b')
ax.scatter([0, 1],[sol2, 0],[1-sol2, 0], c='b', marker='x', depthshade=False)

solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c=(0.7,0.2,0.2), marker = 'o')
# l = ax.legend([solutionSurfaceFakeLine, sol1Line, sol2Line], 
#           [r'$\omega_1,\omega_2,\omega_3$ solution space', r'$\omega_1$, $\omega_2$ partial solution', r'$\omega_2$, $\omega_3$ partial solution'], 
#           numpoints=1, framealpha=0.6)

ax.xaxis.set_ticks([0,0.5,1])
ax.yaxis.set_ticks([0,0.5,1])
ax.zaxis.set_ticks([0,0.5,1])


#========================================================================================#


# Partial solution 1
sol1 = 0.24
ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='g')

# Partial solution 2
sol2 = 0.52
ax.plot([0, 1],[sol2, 0],[1-sol2, 0], linestyle='--', c='b')

# Partial solution plane 1
computed_point1 = np.array([sol1, 1-sol1, 0])
avg_known_points1 = np.array([0,0,1])
relevant_known_point1 = np.array([1,0,0])
to_project1 = relevant_known_point1 - avg_known_points1
to_project_onto1 = computed_point1 - avg_known_points1
projection1 = (np.dot(to_project1, to_project_onto1)/np.dot(to_project_onto1, to_project_onto1)) * to_project_onto1
norm_to_hyperplane1 = to_project1 - projection1
intercept1 = np.dot(norm_to_hyperplane1-avg_known_points1, computed_point1)


# Partial solution plane 2
computed_point2 = np.array([0, sol2, 1-sol2])
avg_known_points2 = np.array([1,0,0])
relevant_known_point2 = np.array([0,1,0])
to_project2 = relevant_known_point2 - avg_known_points2
to_project_onto2 = computed_point2 - avg_known_points2
projection2 = (np.dot(to_project2, to_project_onto2)/np.dot(to_project_onto2, to_project_onto2)) * to_project_onto2
norm_to_hyperplane2 = to_project2 - projection2
intercept2 = np.dot(norm_to_hyperplane2-avg_known_points2, computed_point2)



eq = np.array([[1,1,1],
               norm_to_hyperplane1,
               norm_to_hyperplane2])
so = np.array([1, intercept1, intercept2])
i = np.linalg.solve(eq, so)
resultPoint = ax.scatter(*i, marker='x', c='r', zorder=10)

#========================================================================================#

fig.legend([solutionSurfaceFakeLine, sol1Line, sol2Line, resultPoint], 
           [r'$\omega_1,\omega_2,\omega_3$ solution space', r'$\omega_1$, $\omega_2$ partial solution', r'$\omega_2$, $\omega_3$ partial solution', r'Solution estimate'], 
           numpoints=1, loc='center right', framealpha=0.6, ncol=1, fontsize=14)

#fig.tight_layout()

plt.subplots_adjust(wspace=0, top=1, right=0.75, left=0)

if SAVE_NOT_SHOW:
	plt.savefig("planes.pdf")
else:
	plt.show()


plt.close()