import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt


def solve_continuous_are_iterative(A, B, Q, R, tol=1e-8):
    # integrate Riccati ODE: dP/dt = A'P + PA - PBR^{-1}B'P + Q
    # until steady state (dP/dt -> 0) gives the ARE solution
    n = A.shape[0]
    Rinv = np.linalg.inv(R)

    def riccati_ode(t, p):
        P = p.reshape(n, n)
        Pdot = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
        return Pdot.flatten()

    sol = solve_ivp(riccati_ode, [0, 200], np.zeros(n * n),
                    method='RK45', rtol=tol, atol=tol)
    P = sol.y[:, -1].reshape(n, n)
    return (P + P.T) / 2


def solve_discrete_are_iterative(Ad, Bd, Q, R, tol=1e-8, max_iter=1000):
    # value iteration: P_{k+1} = Q + A'P_k A - A'P_k B (R + B'P_k B)^{-1} B'P_k A
    n = Ad.shape[0]
    P = np.zeros((n, n))
    for _ in range(max_iter):
        P_new = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
        P_new = (P_new + P_new.T) / 2
        if np.linalg.norm(P_new - P) < tol:
            return P_new
        P = P_new
    return P



# system matrices
A = np.array([
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
    [-3.0,  2.0, -2.0,  1.0],
    [ 1.0, -1.0,  0.5, -1.0]
])
B = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.5]
])
n = A.shape[0]

# continuous-time lqr
Q = np.diag([1e2, 1e2, 1e0, 1e0])
R = 1e-2*np.eye(2)
P = solve_continuous_are_iterative(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# initail state and reference
x0 = np.array([0.0, 0.0, 0.0, 0.0])
xref = np.array([0.5, 1.0, 0.0, 0.0])

# simulation
sol = solve_ivp(
    lambda t, x: A @ x + B @ K @ (xref - x),  # closed loop system
    (0.0, 5.0),  # simulation time interval
    x0,  # initial state
    t_eval=np.linspace(0.0, 5.0, 501)
)

# discrete-time system
Ts = 0.05  # sampling time
Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)

# discrete-time lqr
Pd = solve_discrete_are_iterative(Ad, Bd, Q, R)
Kd = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad

# simulation
t = np.arange(0.0, 5.0, Ts)  # simulation time sequence
y = [x0.copy()]  # list of system state
for _ in t[1:]:
    y.append(Ad @ y[-1] + Bd @ Kd @ (xref - y[-1]))
y = np.array(y).T


# figure plot
plt.figure("continuous-time system")
for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    plt.subplot(411+k)
    plt.plot(sol.t, xref[k] * np.ones_like(sol.t), "k--")
    plt.plot(sol.t, sol.y[k, :])
    plt.legend([label+"ref", label])
plt.xlabel("t")
plt.figure("discrete-time system")
for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    plt.subplot(411+k)
    plt.plot(t, xref[k] * np.ones_like(t), "k--")
    plt.plot(t, y[k, :])
    plt.legend([label+"ref", label])
plt.xlabel("t")
plt.show()
