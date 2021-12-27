import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=3)

# constants
c   = 299792458 * 10**-3 # [km/s]
R_E = 6371 # [km]
mu = 3.986004418 * 10**5 # [km^3/s^2]

t = np.loadtxt('t.txt') # epochs - s
t = t - t[0]

C1C = np.loadtxt('C1C.txt') # pseudorange observations - km
PRN = np.loadtxt('GPS_PRN.txt') # PRN of tracked GPS sats

clk_gps = np.loadtxt('clk_gps.txt') # clock correction for gps sats (transmitters) - s

x_gps = np.loadtxt('x_gps.txt') # gps positions (transmitters) - km
y_gps = np.loadtxt('y_gps.txt')
z_gps = np.loadtxt('z_gps.txt')

vx_gps = np.loadtxt('vx_gps.txt') # gps velocities (transmitters) - km/s
vy_gps = np.loadtxt('vy_gps.txt')
vz_gps = np.loadtxt('vz_gps.txt')

x_swc = np.loadtxt('x_swc.txt') # precise swarm C positions (receivers) - km
y_swc = np.loadtxt('y_swc.txt')
z_swc = np.loadtxt('z_swc.txt')

vx_swc = np.loadtxt('vx_swc.txt') # precise swarm C velocities (receivers) - km/s
vy_swc = np.loadtxt('vy_swc.txt')
vz_swc = np.loadtxt('vz_swc.txt')

x_gps_eci = np.loadtxt('x_gps_eci.txt') # gps positions (transmitters) - km
y_gps_eci = np.loadtxt('y_gps_eci.txt')
z_gps_eci = np.loadtxt('z_gps_eci.txt')

vx_gps_eci = np.loadtxt('vx_gps_eci.txt') # gps velocities (transmitters) - km/s
vy_gps_eci = np.loadtxt('vy_gps_eci.txt')
vz_gps_eci = np.loadtxt('vz_gps_eci.txt')

x_swc_eci = np.loadtxt('x_swc_eci.txt') # precise swarm C positions (receivers) - km
y_swc_eci = np.loadtxt('y_swc_eci.txt')
z_swc_eci = np.loadtxt('z_swc_eci.txt')

vx_swc_eci = np.loadtxt('vx_swc_eci.txt') # precise swarm C velocities (receivers) - km/s
vy_swc_eci = np.loadtxt('vy_swc_eci.txt')
vz_swc_eci = np.loadtxt('vz_swc_eci.txt')

clk_swc = np.loadtxt('clk_swc.txt')

# -------------------------- data manipulation ------------------------------
# data corrections
for i, epoch in enumerate(t):

    # light time correction
    tau = C1C[i] / c
    x_gps_eci[i] = np.array(x_gps_eci[i] - tau * vx_gps_eci[i])
    y_gps_eci[i] = np.array(y_gps_eci[i] - tau * vy_gps_eci[i])
    z_gps_eci[i] = np.array(z_gps_eci[i] - tau * vz_gps_eci[i])

    # eccentricity and relativity correction
    dr_cor = -2 / c * (x_gps_eci[i] * vx_gps_eci[i] +
                       y_gps_eci[i] * vy_gps_eci[i] +
                       z_gps_eci[i] * vz_gps_eci[i])
    C1C[i] = C1C[i] + dr_cor


# -------------------------- shooting problem start ------------------------------
# initial guess of x_0
converged = False
x = np.array([-3381.56709263, 152.26868071, -5916.71038428, 6.55999553, -1.01325858, -3.77709814, mu])  # [x,y,z,vx,vy,vz]
x_per =  np.concatenate((np.random.normal(0, 4000, 3),
                         np.random.normal(0, 4, 3),
                         np.random.normal(0, 10**6, 1)))
x = x - x_per
S_scale = np.identity(np.size(x))
S_scale[-1,-1] = 10**6
S_scale[3,3] = 10**-1
S_scale[4,4] = 10**-1
S_scale[5,5] = 10**-1
first = True

while not converged:

    # collection arrays of states
    y  = np.zeros(np.size(t), dtype=object)

    for i, epoch in enumerate(t):

        # initializing
        if i == 0:
            y[i] = np.copy(x[:6])
            Phi = np.identity(np.size(y[i]))
            N_S = np.zeros((np.size(x), np.size(x)))
            b_S = np.zeros(np.size(x))
            S = np.zeros((6, 1))

        # partial design matrix
        H = np.array([(y[i][0] - x_gps_eci[i]) / np.sqrt(
                                (y[i][0] - x_gps_eci[i])**2 +
                                (y[i][1] - y_gps_eci[i])**2 +
                                (y[i][2] - z_gps_eci[i])**2),
                        (y[i][1] - y_gps_eci[i]) / np.sqrt(
                                (y[i][0] - x_gps_eci[i])**2 +
                                (y[i][1] - y_gps_eci[i])**2 +
                                (y[i][2] - z_gps_eci[i])**2),
                        (y[i][2] - z_gps_eci[i]) / np.sqrt(
                                (y[i][0] - x_gps_eci[i])**2 +
                                (y[i][1] - y_gps_eci[i])**2 +
                                (y[i][2] - z_gps_eci[i])**2),
                        np.zeros(np.size(x_gps_eci[i])),
                        np.zeros(np.size(x_gps_eci[i])),
                        np.zeros(np.size(x_gps_eci[i]))])
        H = np.reshape(np.concatenate(H.T), (np.size(x_gps_eci[i]), np.size(y[i])))

        # lsq matrix
        H = np.hstack((H @ Phi, H @ S))
        H_S = H @ S_scale

        # lsq left hand side
        z_est = np.sqrt((y[i][0] - x_gps_eci[i]) ** 2 + (y[i][1] - y_gps_eci[i]) ** 2 +
                      (y[i][2] - z_gps_eci[i]) ** 2) + c * (clk_swc[i] - clk_gps[i])
        dz = C1C[i] - z_est

        # update lsq values
        N_S += H_S.T @ H_S
        b_S += H_S.T @ dz

        # store range for ease of computation
        r2 = np.sum(y[i][:3]**2)

        # y propagation matrix
        f = np.array([y[i][3], y[i][4], y[i][5],
                      (-x[6]*y[i][0])/r2**(3/2),
                      (-x[6]*y[i][1])/r2**(3/2),
                      (-x[6]*y[i][2])/r2**(3/2)])

        # phi propagation matrix
        dfdy = np.zeros((np.size(y[i]), np.size(f)))
        dfdy[:3,3:] = np.identity(3)
        dadr = -x[6] * np.array([[(r2**(3/2) - 3 * y[i][0]**2 * r2**(1/2))/(r2**3), 3*y[i][0]*y[i][1]/r2**(5/2), 3*y[i][0]*y[i][2]/r2**(5/2)],
                               [3*y[i][1]*y[i][0]/r2**(5/2), (r2**(3/2) - 3 * y[i][1]**2 * r2**(1/2))/(r2**3), 3*y[i][1]*y[i][2]/r2**(5/2)],
                               [3*y[i][2]*y[i][0]/r2**(5/2), 3*y[i][2]*y[i][1]/r2**(5/2), (r2**(3/2) - 3 * y[i][2]**2 * r2**(1/2))/ (r2**3)]])
        dfdy[3:,:3] = dadr

        # S propagation matrix
        dfdp = np.array([[0],
                         [0],
                         [0],
                         [-y[i][0]/r2**(3/2)],
                         [-y[i][1]/r2**(3/2)],
                         [-y[i][2]/r2**(3/2)]])

        # propagate y and phi and S
        if i != np.size(t)-1:
            y[i+1] = y[i] + (t[i+1] - t[i])*f
            Phi += (t[i+1] - t[i]) * dfdy @ Phi
            S += (t[i+1] - t[i]) * (dfdy @ S + dfdp)


    if first:
        y_first = np.copy(y)
        first = False

    # solve dx
    dx_S = np.linalg.solve(N_S,b_S)
    dx = S_scale @ dx_S
    print("|dx| = {0:.8f} \nCondition number = {1:.5f}".format(np.linalg.norm(dx), np.linalg.cond(N_S)))

    # check convergence
    if np.linalg.norm(dx) <= 10**-8:
        converged = True


    # iterate
    x += dx

# -------------------------- Plotting and error computation ------------------------------
y = np.concatenate(y).reshape((np.size(t),np.size(y[0])))
y_first = np.concatenate(y_first).reshape((np.size(t),np.size(y_first[0])))

ep = np.sqrt((x_swc_eci - y[:,0])**2 + (y_swc_eci-y[:,1])**2 + (z_swc_eci - y[:,2])**2)*1000
ev = np.sqrt((vx_swc_eci - y[:,3])**2 + (vy_swc_eci-y[:,4])**2 + (vz_swc_eci - y[:,5])**2)*1000
ef = np.sqrt((x_swc_eci - y_first[:,0])**2 + (y_swc_eci-y_first[:,1])**2 + (z_swc_eci - y_first[:,2])**2)*1000

e = np.sqrt((x_swc_eci - y[:,0])**2 +
            (y_swc_eci - y[:,1])**2 +
            (z_swc_eci - y[:,2])**2 +
            (vx_swc_eci - y[:,3])**2 +
            (vy_swc_eci - y[:,4])**2 +
            (vz_swc_eci - y[:,5])**2)

ep_max = np.max(ep)
ev_max = np.max(ev)
ef_max = np.max(ef)

# plot errors
sns.set_theme()
plt.subplot(1,2,1)
plt.plot(t, ef, c = "k", ls = "--")
plt.scatter(t, ef, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Position error: $|\epsilon_p|$ [m]")
plt.title("Position error per epoch for first iteration")
plt.text(t[np.where(ef == ef_max)]-(t[-1]-t[0])*0.2, ef_max+0.0001*ef_max, "Max error {0:.1f} [m]".format(ef_max))
plt.scatter(t[np.where(ef == ef_max)], ef_max, c = "r", zorder=5, s = 25)

sns.set_theme()
plt.subplot(1,2,2)
plt.plot(t, ep, c = "k", ls = "--")
plt.scatter(t, ep, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Position error: $|\epsilon_p|$ [m]")
plt.title("Position error per epoch for final iteration")
plt.text(t[np.where(ep == ep_max)]-(t[-1]-t[0])*0.2, ep_max+0.01*ep_max, "Max error {0:.1f} [m]".format(ep_max))
plt.scatter(t[np.where(ep == ep_max)], ep_max, c = "r", zorder=5, s = 25)

plt.style.use("dark_background")
fig = plt.figure()

# true position plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_facecolor("k")
plt.axis('off')

# true
ax.scatter(x_swc_eci, y_swc_eci, z_swc_eci, c = "r", label="True position")

# Earth
u, v = np.mgrid[-np.pi:np.pi:20j, 0:np.pi:20j]
xe = R_E * np.cos(u)*np.sin(v)
ye = R_E * np.sin(u)*np.sin(v)
ze = R_E * np.cos(v)
ax.plot_wireframe(xe, ye, ze, color="royalblue", linewidth=0.5, zorder=-1)
ax.set_aspect('equal')
plt.legend()


# estimated position plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_facecolor("k")
plt.axis('off')

# estimated
ax.scatter(y[:,0], y[:,1], y[:,2], c = "w", marker="x", label = "Estimated position")

# Earth
ax.plot_wireframe(xe, ye, ze, color="royalblue", linewidth=0.5, zorder=-1)
ax.set_aspect('equal')
plt.legend()
plt.show()