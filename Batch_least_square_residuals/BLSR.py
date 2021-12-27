import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('display.max_columns', None)

c   = 299792458 * 10**-3 # [km/s]
R_E = 6371 # [km]
w_e = 7.292115 * 10**-5 #[rad/s]

PRN      = np.loadtxt('_PRN.txt') # PRN of tracked GPS sats
C1C      = np.loadtxt('C1C.txt') # pseudorange observations - km
clk_gps  = np.loadtxt('clk_gps.txt') # clock correction for gps sats (transmitters) - s
t        = np.loadtxt('t.txt') # epochs - s
vx_gps   = np.loadtxt('vx_gps.txt') # gps velocities (transmitters) - km/s
vy_gps   = np.loadtxt('vy_gps.txt')
vz_gps   = np.loadtxt('vz_gps.txt')
vx_swc   = np.loadtxt('vx_swc.txt') # precise swarm C velocities (receivers) - km/s
vy_swc   = np.loadtxt('vy_swc.txt')
vz_swc   = np.loadtxt('vz_swc.txt')
x_gps    = np.loadtxt('x_gps.txt') # gps positions (transmitters) - km
y_gps    = np.loadtxt('y_gps.txt')
z_gps    = np.loadtxt('z_gps.txt')
x_swc    = np.loadtxt('x_swc.txt') # precise swarm C positions (receivers) - km
y_swc    = np.loadtxt('y_swc.txt')
z_swc    = np.loadtxt('z_swc.txt')

# storage array for lsq data
x_swc_lsq = np.zeros(np.size(x_swc))
y_swc_lsq = np.zeros(np.size(y_swc))
z_swc_lsq = np.zeros(np.size(z_swc))
residuals = np.zeros(np.size(t))

data = np.zeros(np.shape(PRN)[0], dtype=object)

for i, epoch in enumerate(tqdm(t)):

    # data formatting
    frame = np.array([[j for j in PRN[i] if j != 0],
                     [j for j in C1C[i] if j != 0],
                     [j for j in clk_gps[i] if j != 0],
                     [j for j in x_gps[i] if j != 0],
                     [j for j in y_gps[i] if j != 0],
                     [j for j in z_gps[i] if j != 0],
                     [j for j in vx_gps[i] if j != 0],
                     [j for j in vy_gps[i] if j != 0],
                     [j for j in vz_gps[i] if j != 0]])

    data[i] = pd.DataFrame(frame.T, columns = ["identifier", "pseudorange",
                                               "clock_corr", "x_gps", "y_gps",
                                               "z_gps", "vx_gps","vy_gps", "vz_gps"])

    # range corrections
    data[i]["tau"] = data[i].pseudorange/c
    gamma = data[i]["tau"]*w_e
    data[i]["x_gps_corr"] = np.cos(gamma)*(data[i].x_gps - data[i]["tau"] * data[i].vx_gps) + \
                            np.sin(gamma)*(data[i].y_gps - data[i]["tau"] * data[i].vy_gps)
    data[i]["y_gps_corr"] = -np.sin(gamma)*(data[i].x_gps - data[i]["tau"] * data[i].vx_gps) + \
                            np.cos(gamma)*(data[i].y_gps - data[i]["tau"] * data[i].vy_gps)
    data[i]["z_gps_corr"] = data[i].z_gps - data[i]["tau"] * data[i].vz_gps

    # eccentricity and orbit variation corrections
    data[i]["dr_rel"] = - 2/c * (data[i].x_gps_corr * data[i].vx_gps +
                                 data[i].y_gps_corr * data[i].vy_gps +
                                 data[i].z_gps_corr * data[i].vz_gps)
    data[i]["pseudorange_corr"] = data[i].pseudorange + data[i].dr_rel

    x = np.zeros(4) # [rr_x, rr_y, rr_z, dtr]

    S = np.identity(4)
    S[np.size(x) - 1, np.size(x) - 1] = 10 ** -6

    converged = False
    while not converged:

        # setup of lsq regression matrices and vectors
        A_x = np.sqrt((x[0] - data[i].x_gps_corr)**2 + (x[1] - data[i].y_gps_corr)**2 +
                      (x[2] - data[i].z_gps_corr)**2) + c * (x[3] - data[i].clock_corr)

        H_x = np.array([(x[0] - data[i].x_gps_corr) / np.sqrt(
                                (x[0] - data[i].x_gps_corr)**2 +
                                (x[1] - data[i].y_gps_corr)**2 +
                                (x[2] - data[i].z_gps_corr)**2),
                        (x[1] - data[i].y_gps_corr) / np.sqrt(
                                (x[0] - data[i].x_gps_corr)**2 +
                                (x[1] - data[i].y_gps_corr)**2 +
                                (x[2] - data[i].z_gps_corr)**2),
                        (x[2] - data[i].z_gps_corr) / np.sqrt(
                                (x[0] - data[i].x_gps_corr)**2 +
                                (x[1] - data[i].y_gps_corr)**2 +
                                (x[2] - data[i].z_gps_corr)**2),
                        (c*np.ones(np.size(data[i].x_gps_corr)))])

        H_x = np.reshape(np.concatenate(H_x.T), (np.size(data[i].x_gps_corr), np.size(x)))
        dy = data[i].pseudorange_corr - A_x

        # scale H for better condition number of N
        H_x_s = H_x @ S

        # solve for dx
        N_x_s = H_x_s.T @ H_x_s

        b_x_s = H_x_s.T @ dy
        dx_s = np.linalg.solve(N_x_s, b_x_s)
        dx = S @ dx_s

        # adjust x and iterate
        x = x + dx

        # check convergence
        cond_N_x_s = np.linalg.cond(N_x_s)

        if np.linalg.norm(dx) <= 10**-7 and cond_N_x_s <= 350:
            converged = True
            x_swc_lsq[i] = x[0]
            y_swc_lsq[i] = x[1]
            z_swc_lsq[i] = x[2]
            residuals[i] = np.linalg.norm(H_x @ dx - dy)

# errors
e = np.sqrt((x_swc - x_swc_lsq)**2+(y_swc - y_swc_lsq)**2+(z_swc - z_swc_lsq)**2)*1000
e_max = np.max(e)

# plot errors
sns.set_theme()
plt.plot(t, e, c = "k", ls = "--")
plt.scatter(t, e, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Error: $|\epsilon|$ [m]")
plt.title("Error per epoch")
plt.text(t[np.where(e == e_max)], e_max+0.2, "Max error {0:.1f} [m]".format(e_max))
plt.scatter(t[np.where(e == e_max)], e_max, c = "r", zorder=5, s = 25)

plt.style.use("dark_background")
fig = plt.figure()

# true position plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_facecolor("k")
plt.axis('off')

# true
ax.scatter(x_swc, y_swc, z_swc, c = "r", label="True position")

# Earth
u, v = np.mgrid[-np.pi:np.pi:20j, 0:np.pi:20j]
x = R_E * np.cos(u)*np.sin(v)
y = R_E * np.sin(u)*np.sin(v)
z = R_E * np.cos(v)
ax.plot_wireframe(x, y, z, color="royalblue", linewidth=0.5, zorder=-1)
ax.set_aspect('equal')
plt.legend()

# estimated position plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_facecolor("k")
plt.axis('off')

# estimated
ax.scatter(x_swc_lsq, y_swc_lsq, z_swc_lsq, c = "w", marker="x", label = "Estimated position")

# Earth
u, v = np.mgrid[-np.pi:np.pi:20j, 0:np.pi:20j]
x = R_E * np.cos(u)*np.sin(v)
y = R_E * np.sin(u)*np.sin(v)
z = R_E * np.cos(v)
ax.plot_wireframe(x, y, z, color="royalblue", linewidth=0.5, zorder=-1)
ax.set_aspect('equal')
plt.legend()
plt.show()