import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import matplotlib.tri as mtri
import numpy as np
import sympy as sp


#========================================================================

# Plotting helpers, from https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals) # eigvals positive because covariance is positive semi definite
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)



def omega_exact(sensor_traces):
    inv_sum = sum((1/x for x in sensor_traces))
    weights = []
    for trace in sensor_traces:
        weights.append((1/trace)/inv_sum)
    return weights


def omega_estimates(sensor_lists, step):
    num_sensors = len(sensor_lists)
    norm_to_solution_hyperplane = np.array([1]*num_sensors)

    sensor_hyperplanes = []
    sensor_hyperplanes_eqns = [norm_to_solution_hyperplane]
    sensor_hyperplanes_intercepts = [1]

    # n sensors to loop
    for i in range(num_sensors-1):
        hyperplane_points = []

        # Get all partial solution hyperplane points that are known exactly. n^2 run time as that's the number of coordinates
        for j in range(num_sensors):
            if j == i or j == i+1:
                continue
            known_point = np.array([0]*num_sensors)
            known_point[j] = 1
            hyperplane_points.append(known_point)
        
        # Find the remaing unknown partial solution hyperplane point - when all omega values (except for current and next sensor) are 0. log(p) run time.
        list_a = sensor_lists[i]
        list_b = list(reversed(sensor_lists[i+1]))
        om = intersect_approx_bsearch(list_a, list_b, step)
        computed_point = np.array([0]*i + [om, 1-om] + [0]*(num_sensors-2-i))

        # Special case with 2 sensors, the computed point is in fact the solution list of omegas
        if num_sensors==2:
            return computed_point

        # Get the normal to the hyperplane defined by all the known points, and the computed point above
        avg_known_points = np.mean(hyperplane_points, axis=0)
        relevant_known_point = np.array([0]*i + [1] + [0]*(num_sensors-1-i))
        to_project = relevant_known_point - avg_known_points
        to_project_onto = computed_point - avg_known_points
        projection = (np.dot(to_project, to_project_onto)/np.dot(to_project_onto, to_project_onto)) * to_project_onto
        norm_to_hyperplane = to_project - projection

        # # Some debug plotting
        # if PLOT and i==0:
        #     ax.plot(*zip(avg_known_points, relevant_known_point), c='b')
        #     ax.scatter(*avg_known_points, marker='^', c='b')

        #     ax.plot(*zip(avg_known_points, computed_point), c='r')
        #     ax.scatter(*computed_point, marker='^', c='r')
            
        #     ax.plot(*zip(computed_point, computed_point+norm_to_hyperplane), c='g')
        #     ax.scatter(*computed_point+norm_to_hyperplane, marker='^', c='g')

        # Add the computed point to the partial solution hyperplane point list, completing the list
        hyperplane_points.append(computed_point)

        # Compute the hyperplane equation in the form ax1 + bx2 + ... + intercept = 0. Store as vector ((a,b,...), intercept) for easier computing later
        intercept = np.dot(norm_to_hyperplane, computed_point)
        sensor_hyperplanes_intercepts.append(intercept)
        sensor_hyperplanes_eqns.append(norm_to_hyperplane)

        # Add to list of sensor planes
        sensor_hyperplanes.append(np.array(hyperplane_points))
    
    # Convert all hyperplane equations to numpy arrays for solving
    sensor_hyperplanes_eqns = np.array(sensor_hyperplanes_eqns)
    sensor_hyperplanes_intercepts = np.array(sensor_hyperplanes_intercepts).T

    # TODO Should handle case of multiple 0 traces, by equally weighting them all and making the rest 0
    # Solve intersection of all hyperplanes
    omegas = np.linalg.solve(sensor_hyperplanes_eqns, sensor_hyperplanes_intercepts)

    # Some debug printing and plotting
    #print(sensor_hyperplanes_eqns)
    #print(sensor_hyperplanes_intercepts)
    #print(sensor_hyperplanes)

    # if PLOT:
    #     ax.scatter(*zip(*[(a,b,c) for a in np.arange(0,1.1,0.1) for b in np.arange(0,1.1,0.1) for c in np.arange(0,1.1,0.1) if np.isclose(a+b+c, 1)]), c='grey')
    #     plt.show()

    return omegas

def intersect_approx(increasing_cmp_list, decreasing_cmp_list, step):
    curr_om = 0
    found_om = 0
    for i in range(len(increasing_cmp_list)):
        if increasing_cmp_list[i] == decreasing_cmp_list[i]:
            found_om = curr_om
            break
        elif increasing_cmp_list[i] > decreasing_cmp_list[i]:
            found_om = curr_om - 0.5*step
            break
        curr_om += step
    
    # Debug printing
    # print('approx intersection')
    # print(['%1.4f'%i for i in increasing_cmp_list])
    # print(['%1.4f'%i for i in decreasing_cmp_list])
    # print(found_om)
    # print('exact intersection')
    # print(intersect_exact(increasing_cmp_list, decreasing_cmp_list, step))
    # print()

    return found_om

# Faster version of the above
def intersect_approx_bsearch(l1, l2, discStep):
    # l1 < l2 to start with always
    startDiff = True
        
    interceptInd, approx = listIntersectSup(l1, l2, 0, startDiff)
    if approx:
        return (interceptInd - 0.5)*discStep
    else:
        return interceptInd*discStep

def listIntersectSup(l1, l2, index, startDiff):
    n = len(l1)

    # Base case
    if n == 1:
        if l1[0] == l2[0]:
            return (index, False)
        elif (l1[0] < l2[0]) == startDiff:
            return (index + 1, True)
        elif (l1[0] < l2[0]) != startDiff:
            return (index, True)
    
    # Single recurse go left or right depending on whether the sign has changed
    mid = n//2
    if l1[mid] == l2[mid]:
        return (index + mid, False)
    elif (l1[mid] < l2[mid]) == startDiff:
        return listIntersectSup(l1[mid:], l2[mid:], index+mid, startDiff)
    elif (l1[mid] < l2[mid]) != startDiff:
        return listIntersectSup(l1[:mid], l2[:mid], index, startDiff)



def intersect_exact(increasing_cmp_list, decreasing_cmp_list, step):
    # Don't need step for exact solution, pass it to keep signature akin to the approximation function
    inc = increasing_cmp_list[-1]
    dec = decreasing_cmp_list[0]
    return dec/(inc+dec)

#============================================================================


SAVE_NOT_SHOW = True

if SAVE_NOT_SHOW:
	mpl.use("pgf")

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",   
    'font.family': 'serif',         # Use serif/main font for text elements
    'text.usetex': True,            # Use inline maths for ticks
    'pgf.rcfonts': False,           # Don't setup fonts from matplotlib rc params
})

#=================================================================

# Sim data
sim_data = pkl.load(open("simout.p", "rb"))
# Sim pic params
MAX_STEPS = 30
SKIP_FIRST = 0
TIME_BETWEEN_PLOTS = 3

# Trace pic params
TRACE_LIMIT_POINTS = 50

# Omega pic params
OMEGA_LIMIT_POINTS = 50

#=================================================================

fig = plt.figure()
fig.set_size_inches(w=4, h=3)

#=================================================================


ax = fig.add_subplot(111)
#ax.set_title(r'FCI and SecFCI estimate traces')

# FCI estimates
split = list(zip(*sim_data['fusion_estimates']))
estimates = split[0]
errors = split[1]
error_traces = [np.trace(p) for p in errors][:TRACE_LIMIT_POINTS]
ax.plot(error_traces, c='r', label=r'Fusion trace (FCI)', marker='.')


# Secure FCI estimates
split = list(zip(*sim_data['secure_fusion_estimates']))
estimates = split[0]
errors = split[1]
error_traces = [np.trace(p) for p in errors][:TRACE_LIMIT_POINTS]
ax.plot(error_traces, c='b', label=r'Fusion trace (our method)', marker='.', linestyle='')

plt.xlabel(r'Simulation Time')
plt.legend()
plt.tight_layout()


#=================================================================

if SAVE_NOT_SHOW:
	plt.savefig("sim.pdf")
else:
	plt.show()


#plt.close()


#=================================================================



# Ensure this is the last plot as this fucks up the data (could copy it but meh)
gf_true = sim_data['ground_truth']
sim_data['ground_truth'] = [x for i,x in enumerate(sim_data['ground_truth']) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['measurements'][0] = [x for i,x in enumerate(sim_data['measurements'][0]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['measurements'][1] = [x for i,x in enumerate(sim_data['measurements'][1]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['measurements'][2] = [x for i,x in enumerate(sim_data['measurements'][2]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['sensor_estimates'][0] = [x for i,x in enumerate(sim_data['sensor_estimates'][0]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['sensor_estimates'][1] = [x for i,x in enumerate(sim_data['sensor_estimates'][1]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['sensor_estimates'][2] = [x for i,x in enumerate(sim_data['sensor_estimates'][2]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['fusion_estimates'] = [x for i,x in enumerate(sim_data['fusion_estimates']) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['secure_fusion_estimates'] = [x for i,x in enumerate(sim_data['secure_fusion_estimates']) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]



fig = plt.figure()
fig.set_size_inches(w=4, h=3)
ax = fig.add_subplot(111)
#ax.set_title(r'FCI and SecFCI comparison')

# Ground truth
ax.plot(*zip(*[(x[0],x[2]) for x in gf_true[SKIP_FIRST*TIME_BETWEEN_PLOTS:(SKIP_FIRST+MAX_STEPS)*TIME_BETWEEN_PLOTS]]), c='lightgrey', marker='.', zorder=1)

# Measurements 1
ax.scatter(*zip(*[(x[0],x[1]) for x in sim_data['measurements'][0]]), c='limegreen', marker='x', label=r'Sensor 1', zorder=2)
# Lines from gt to measurements
m1 = list(zip(sim_data['ground_truth'], sim_data['measurements'][0]))
for i in range(len(m1)):
    ax.plot([m1[i][0][0], m1[i][1][0]], [m1[i][0][2], m1[i][1][1]], c='lightgrey', linestyle='--', zorder=3)

# Measurements 2
ax.scatter(*zip(*[(x[0],x[1]) for x in sim_data['measurements'][1]]), c='cornflowerblue', marker='x', label=r'Sensor 2', zorder=2)
# Lines from gt to measurements
m2 = list(zip(sim_data['ground_truth'], sim_data['measurements'][1]))
for i in range(len(m2)):
    ax.plot([m2[i][0][0], m2[i][1][0]], [m2[i][0][2], m2[i][1][1]], c='lightgrey', linestyle='--', zorder=3)

# Measurements 3
ax.scatter(*zip(*[(x[0],x[1]) for x in sim_data['measurements'][2]]), c='orange', marker='x', label=r'Sensor 3', zorder=2)
# Lines from gt to measurements
m3 = list(zip(sim_data['ground_truth'], sim_data['measurements'][2]))
for i in range(len(m3)):
    ax.plot([m3[i][0][0], m3[i][1][0]], [m3[i][0][2], m3[i][1][1]], c='lightgrey', linestyle='--', zorder=3)

# FCI estimates
# split = list(zip(*sim_data['fusion_estimates']))
# estimates = split[0]
# errors = split[1]

# estimates2D = [np.array([e[0],e[2]]) for e in estimates]
# errors2D = [np.array([[p[0,0], p[2,0]], [p[0,2], p[2,2]]]) for p in errors]

# ax.scatter(None, None, c='r', marker='.', label=r'FCI estimate', zorder=4)
# for i in range(len(estimates)):
#     estimate = estimates2D[i]
#     error = errors2D[i]

#     ax.scatter(*estimate, c='r', marker='.')
#     ax.add_artist(ph.get_cov_ellipse(error, estimate, 2, fill=False, linestyle='-', edgecolor='r', zorder=4))


# Secure FCI estimates
split = list(zip(*sim_data['secure_fusion_estimates']))
estimates = split[0]
errors = split[1]

estimates2D = [np.array([e[0],e[2]]) for e in estimates]
errors2D = [np.array([[p[0,0], p[2,0]], [p[0,2], p[2,2]]]) for p in errors]

ax.scatter(None, None, c='b', marker='.', label=r'Fused estimate', zorder=4)
for i in range(len(estimates)):
    estimate = estimates2D[i]
    error = errors2D[i]

    ax.scatter(*estimate, c='b', marker='.')
    ax.add_artist(get_cov_ellipse(error, estimate, 2, fill=False, linestyle='-', edgecolor='b', zorder=4))

plt.xlabel(r'Location $x$')
plt.ylabel(r'Location $y$')

# Move the legend up slightly
ax.xaxis.labelpad = 0

plt.legend()
plt.tight_layout()


#=================================================================

if SAVE_NOT_SHOW:
  plt.savefig("sim_run.pdf")
else:
  plt.show()


plt.close()





#=================================================================


fig = plt.figure()
fig.set_size_inches(w=4, h=3)
ax = fig.add_subplot(111)
#ax.set_title(r'Difference in $\omega_i$ values')

split1 = list(zip(*sim_data['sensor_estimates'][0]))
errors1 = split1[1][:OMEGA_LIMIT_POINTS]
error_traces1 = [np.trace(p) for p in errors1][:OMEGA_LIMIT_POINTS]

split2 = list(zip(*sim_data['sensor_estimates'][1]))
errors2 = split2[1][:OMEGA_LIMIT_POINTS]
error_traces2 = [np.trace(p) for p in errors2][:OMEGA_LIMIT_POINTS]

split3 = list(zip(*sim_data['sensor_estimates'][2]))
errors3 = split3[1][:OMEGA_LIMIT_POINTS]
error_traces3 = [np.trace(p) for p in errors3][:OMEGA_LIMIT_POINTS]

trace_groups = list(zip(error_traces1, error_traces2, error_traces3))
omegas = [omega_exact(ts) for ts in trace_groups]
omega_step_size = 0.1
approx_omegas = [omega_estimates([[w*i for w in np.arange(0, 1+omega_step_size, omega_step_size)] for i in ts], omega_step_size) for ts in trace_groups]

split_omegas = list(zip(*omegas))
split_approx_omegas = list(zip(*approx_omegas))

fci1, = ax.plot([i*0.5 for i in range(len(split_omegas[0]))], split_omegas[0], c=(0.9,0,0), marker='.')
fci2, = ax.plot([i*0.5 for i in range(len(split_omegas[1]))], split_omegas[1], c=(0.9,0.2,0.2), marker='.')
fci3, = ax.plot([i*0.5 for i in range(len(split_omegas[2]))], split_omegas[2], c=(0.9,0.4,0.4), marker='.')

secFci1, = ax.plot([i*0.5 for i in range(len(split_approx_omegas[0]))], split_approx_omegas[0], c=(0,0,0.9), marker='.')
secFci2, = ax.plot([i*0.5 for i in range(len(split_approx_omegas[1]))], split_approx_omegas[1], c=(0.2,0.2,0.9), marker='.')
secFci3, = ax.plot([i*0.5 for i in range(len(split_approx_omegas[2]))], split_approx_omegas[2], c=(0.4,0.4,0.9), marker='.')


diff = np.sqrt((np.array(split_omegas[0]) - np.array(split_approx_omegas[0]))**2 + \
               (np.array(split_omegas[1]) - np.array(split_approx_omegas[1]))**2 + \
               (np.array(split_omegas[2]) - np.array(split_approx_omegas[2]))**2)
er, = ax.plot([i*0.5 for i in range(len(diff))], diff, c='grey', marker='.')
erbound, = ax.plot([i*0.5 for i in range(len(diff))], [np.sqrt(3*(omega_step_size/2.0)**2) for i in range(len(diff))], c='lightskyblue', marker='', linestyle='--')

ax.legend([(secFci1,secFci2,secFci3),(fci1,fci2,fci3),er,erbound], [r'$\omega_i^{(SecFCI)}$', r'$\omega_i^{(FCI)}$', r'$|\underline{\omega}^{(FCI)}-\underline{\omega}^{(SecFCI)}|$', r'Error Bound'], 
          numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='lower center')

plt.xlabel(r'Simulation Time')
plt.ylabel(r'Values of $\omega_i$')


plt.tight_layout()
if SAVE_NOT_SHOW:
    plt.savefig('sim_omegas.pdf')
else:
    plt.show()
plt.close()