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

# -------------------------- kalman filter  ------------------------------
# initial guess of x_0
x = np.array([-3381.56709263, 152.26868071, -5916.71038428, 6.55999553, -1.01325858, -3.77709814])  # [x,y,z,vx,vy,vz]
x_per = 0.5/np.sqrt(3)
# x = x - x_per

# collection arrays of states
y  = np.zeros(np.size(t), dtype=object)
y[0] = x

# covariance matrices initialization and collection arrays
R = np.identity(8) * (5 * 10**-3)**2
P0 = np.identity(6) * (10 * 10**-3)**2
P = np.zeros(np.size(t), dtype=object)
P[0] = P0

# start iterating over epochs
for i, epoch in enumerate(t):

    # partial design matrix
    H = np.array([(y[i][0] - x_gps_eci[i]) / np.sqrt(
                      (y[i][0] - x_gps_eci[i]) ** 2 +
                      (y[i][1] - y_gps_eci[i]) ** 2 +
                      (y[i][2] - z_gps_eci[i]) ** 2),
                   (y[i][1] - y_gps_eci[i]) / np.sqrt(
                      (y[i][0] - x_gps_eci[i]) ** 2 +
                      (y[i][1] - y_gps_eci[i]) ** 2 +
                      (y[i][2] - z_gps_eci[i]) ** 2),
                   (y[i][2] - z_gps_eci[i]) / np.sqrt(
                      (y[i][0] - x_gps_eci[i]) ** 2 +
                      (y[i][1] - y_gps_eci[i]) ** 2 +
                      (y[i][2] - z_gps_eci[i]) ** 2),
                   np.zeros(np.size(x_gps_eci[i])),
                   np.zeros(np.size(x_gps_eci[i])),
                   np.zeros(np.size(x_gps_eci[i]))])
    H = np.reshape(np.concatenate(H.T), (np.size(x_gps_eci[i]), np.size(y[i])))

    #observation and state deviations
    h_x = np.sqrt((y[i][0] - x_gps_eci[i]) ** 2 + (y[i][1] - y_gps_eci[i]) ** 2 +
                    (y[i][2] - z_gps_eci[i]) ** 2) + c * (clk_swc[i] - clk_gps[i])
    dz = C1C[i] - h_x

    # Kalman gain
    K = P[i] @ H.T @ np.linalg.inv((H @ P[i] @ H.T + R))

    # Kalman updates
    y[i] += K @ dz
    P[i] = (np.identity(6) - K @ H) @ P[i]

    # store range for ease of computation
    r2 = np.sum(y[i][:3] ** 2)

    # initialize Phi
    Phi = np.identity(6)

    # y propagation matrix
    f = np.array([y[i][3], y[i][4], y[i][5],
                  (-mu * y[i][0]) / np.sum(y[i][:3] ** 2) ** (3 / 2),
                  (-mu * y[i][1]) / np.sum(y[i][:3] ** 2) ** (3 / 2),
                  (-mu * y[i][2]) / np.sum(y[i][:3] ** 2) ** (3 / 2)])

    # phi propatation matrix
    dfdy = np.zeros((np.size(y[i]), np.size(f)))
    dfdy[:3, 3:] = np.identity(3)
    dadr = -mu * np.array([[(r2 ** (3 / 2) - 3 * y[i][0] ** 2 * r2 ** (1 / 2)) / (r2 ** 3),
                            3 * y[i][0] * y[i][1] / r2 ** (5 / 2), 3 * y[i][0] * y[i][2] / r2 ** (5 / 2)],
                           [3 * y[i][1] * y[i][0] / r2 ** (5 / 2),
                            (r2 ** (3 / 2) - 3 * y[i][1] ** 2 * r2 ** (1 / 2)) / (r2 ** 3),
                            3 * y[i][1] * y[i][2] / r2 ** (5 / 2)],
                           [3 * y[i][2] * y[i][0] / r2 ** (5 / 2), 3 * y[i][2] * y[i][1] / r2 ** (5 / 2),
                            (r2 ** (3 / 2) - 3 * y[i][2] ** 2 * r2 ** (1 / 2)) / (r2 ** 3)]])
    dfdy[3:, :3] = dadr

    # propagate y, Phi and covariance matrix
    if i != np.size(t) - 1:
        y[i + 1] = np.copy(y[i] + (t[i + 1] - t[i]) * f)
        Phi += (t[i + 1] - t[i]) * dfdy @ Phi
        P[i + 1] = Phi @ P[i] @ Phi.T + 10**-4 * P0

# -------------------------- Plotting and error computation ------------------------------
y = np.concatenate(y).reshape((np.size(t),np.size(y[0])))

ep = np.sqrt((x_swc_eci - y[:,0])**2 + (y_swc_eci-y[:,1])**2 + (z_swc_eci - y[:,2])**2)*1000
ev = np.sqrt((vx_swc_eci - y[:,3])**2 + (vy_swc_eci-y[:,4])**2 + (vz_swc_eci - y[:,5])**2)*1000

e = np.sqrt((x_swc_eci - y[:,0])**2 +
            (y_swc_eci - y[:,1])**2 +
            (z_swc_eci - y[:,2])**2 +
            (vx_swc_eci - y[:,3])**2 +
            (vy_swc_eci - y[:,4])**2 +
            (vz_swc_eci - y[:,5])**2)

ep_max = np.max(ep)
ev_max = np.max(ev)

# -------------------------- Plotting and variance computation ------------------------------
P = np.concatenate(P).reshape((np.size(t),np.shape(P[0])[0], np.shape(P[0])[1]))

Pp = np.sqrt(P[:,0,0] + P[:,1,1], P[:,2,2]) * 1000
Pv = np.sqrt(P[:,3,3] + P[:,4,4], P[:,5,5]) * 1000

# -------------------------- Plotting errors ------------------------------
sns.set_theme()
fig = plt.figure(0)

plt.subplot(2,2,1)
plt.plot(t, ep, c = "k", ls = "--")
plt.scatter(t, ep, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Position error: $|\epsilon_p|$ [m]")
plt.title("Position error per epoch")
plt.text(t[np.where(ep == ep_max)]-(t[-1]-t[0])*0.2, ep_max+0.01*ep_max, "Max. error {0:.1f} [m]".format(ep_max))
plt.scatter(t[np.where(ep == ep_max)], ep_max, c = "r", zorder=5, s = 25)

plt.subplot(2,2,2)
plt.plot(t, ev, c = "k", ls = "--")
plt.scatter(t, ev, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Velocity error: $|\epsilon_v|$ [m/s]")
plt.title("Velocity error per epoch")
plt.text(t[np.where(ev == ev_max)]-(t[-1]-t[0])*0.2, ev_max+0.01*ev_max, "Max. error {0:.1f} [m/s]".format(ev_max))
plt.scatter(t[np.where(ev == ev_max)], ev_max, c = "r", zorder=5, s = 25)

plt.subplot(2,2,3)
plt.xlim(0,100)
plt.ylim(2.5,4.5)
plt.plot(t, ep, c = "k", ls = "--")
plt.scatter(t, ep, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Position error: $|\epsilon_p|$ [m]")
plt.text(t[np.where(ep == ep_max)]-(t[-1]-t[0])*0.2, ep_max+0.01*ep_max, "Max. error {0:.1f} [m]".format(ep_max))
plt.scatter(t[np.where(ep == ep_max)], ep_max, c = "r", zorder=5, s = 25)

plt.subplot(2,2,4)
plt.xlim(0,100)
plt.ylim(4.2,4.6)
plt.plot(t, ev, c = "k", ls = "--")
plt.scatter(t, ev, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Velocity error: $|\epsilon_v|$ [m/s]")
plt.text(t[np.where(ev == ev_max)]-(t[-1]-t[0])*0.2, ev_max+0.01*ev_max, "Max. error {0:.1f} [m/s]".format(ev_max))
plt.scatter(t[np.where(ev == ev_max)], ev_max, c = "r", zorder=5, s = 25)

# -------------------------- Plotting variances ------------------------------
fig = plt.figure(1)

plt.subplot(1,2,1)
plt.plot(t, Pp, c = "k", ls = "--")
plt.scatter(t, Pp, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Position variance: $|\sigma_p|$ [m]")
plt.title("Position variance per epoch")

plt.subplot(1,2,2)
plt.plot(t, Pv, c = "k", ls = "--")
plt.scatter(t, Pv, c = "k", marker ="x", s = 20)
plt.xlabel("Time: t [s]")
plt.ylabel(r"Velocity variance: $|\sigma_v|$ [m/s]")
plt.title("Velocity variance per epoch")

# -------------------------- Plotting positions ------------------------------
plt.style.use("dark_background")
fig = plt.figure(2)

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
