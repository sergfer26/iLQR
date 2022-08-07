'''
 Quadrotor dynamics
'''

import sympy as sp
import numpy as np

from ilqr import iLQR
from ilqr.utils import GetSyms, Constrain
from ilqr.containers import Dynamics, Cost

from numpy import cos as c
from numpy import sin as s

from sympy.vector import CoordSys3D
from scipy.spatial.transform import Rotation

from numpy.linalg import norm 

C = CoordSys3D('C')

#params
dt = 0.01

G = 9.81
Ixx, Iyy, Izz = 4.856 * 1e-3, 4.856 * 1e-3, 8.801 * 1e-3  # (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140 * 1e-6, 1.433, 0.225
K = 0.001219 #2.980 * 1e-6  # kt
omega_0 = np.sqrt((G * M)/(4 * K))
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0
n_x, n_u = 12, 4

C1, C2, C3 = 0.25, 0.1, 0.005

omega0_per = .60
VEL_MAX = omega_0 * omega0_per  # 60 #Velocidad maxima de los motores 150
VEL_MIN = - omega_0 * omega0_per


def rotation_matrix(angles):
    '''
        rotation_matrix obtine la matriz de rotación de los angulos de Euler
        https://en.wikipedia.org/wiki/Rotation_matrix;

        angles: son los angulos de Euler psi, theta y phi con respecto a los
        ejes x, y, z;

        regresa: la matriz R.
    '''
    z, y, x = angles  # psi, theta, phi
    R = sp.Matrix([
        sp.cos(z) * sp.cos(y), sp.cos(z) * sp.sin(y) * sp.sin(x) - sp.sin(z)
         * sp.cos(x),
         sp.cos(z) * sp.sin(y) * sp.cos(x) + sp.sin(z) * sp.sin(x),
        sp.sin(z) * sp.cos(y), sp.sin(z) * sp.cos(y) * sp.sin(x) + sp.cos(z)
         * sp.cos(x),
         sp.sin(z) * sp.sin(y) * sp.cos(x) - sp.cos(z) * sp.sin(x),
        - sp.sin(y), sp.cos(y) * sp.sin(x), sp.cos(y) * sp.cos(x)
    ])
    return R

#Quadcopter dynamics
def f(state, actions):  # Sistema dinámico
    '''
        f calcula el vector dot_x = f(x, t, w) (sistema dinamico);

        X: es un vector de 12 posiciones;
        t: es un intervalo de tiempo [t1, t2];
        wi: es un parametro de control, i = 1, 2, 3, 4;

        regresa dot_x
    '''
    u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    # Ixx, Iyy, Izz = I
    w1, w2, w3, w4 = actions
    du = r * v - q * w - G * np.sin(theta)
    dv = p * w - r * u - G * np.cos(theta) * np.sin(phi)
    dw = q * u - p * v + G * np.cos(phi) * np.cos(theta) - (K/M) * (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2)
    dp = ((L * B) / Ixx) * (w4 ** 2 - w2 ** 2) - q * r * ((Izz - Iyy) / Ixx)
    dq = ((L * B) / Iyy) * (w3 ** 2 - w1 ** 2) - p * r * ((Ixx - Izz) / Iyy)
    dr = (B/Izz) * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
    dpsi = (q * np.sin(phi) + r * np.cos(phi)) * (1.0 / np.cos(theta))
    dtheta = q * np.cos(phi) - r * np.sin(phi)
    dphi = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    dx = u
    dy = v
    dz = w
    return np.array([du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi]) 


unos = sp.Matrix([1, 0, 0, 0, 1, 0, 0, 0, 1])

state, action = GetSyms(n_x, n_u)
#state_dot = sp.Matrix([0.0]*n_x)
# state_dot = f(state, action) 
pos = state[3: 6]
angles = state[9:12]
R = rotation_matrix(angles)
mat = sp.flatten(unos - R)
cost = C1 * (pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) # C1 * 
cost += C3 * (mat[0] ** 2 + mat[1] ** 2 + mat[2] ** 2 + mat[3] ** 2 + mat[4] ** 2 + mat[5] ** 2 + mat[6] ** 2 + mat[7] ** 2 + mat[8] ** 2)
cost +=  C2 * (action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2)#sp.Matrix(action).norm() #C3 * 

# penalty = sympy_penalty(state, action)
# final_penalty = sympy_penalty_final(state)

#Quadrotor = Dynamics.Continuous(f, dt)
Quadrotor = Dynamics.Continuous(f, dt=dt)
cost = Cost.Symbolic(cost, 0, state, action)
controller = iLQR(Quadrotor, cost)

N = 300
x0 = np.zeros(n_x)
us_init = np.random.uniform(low=VEL_MIN, high=VEL_MAX, size=(N, n_u))#np.random.randn(N, n_u)*0.0001
  
xs, us, cost_trace = controller.fit(x0, us_init, N)

import pandas as pd 
import matplotlib.pyplot as plt

data = pd.DataFrame(xs, columns=['u', 'v', 'w', 'x', 'y', 'z', 'p', 'q', 'r', '$\psi$', '$\\theta$', '$\phi$'])
data.plot(subplots=True, layout=(6, 2))
plt.show()

actions = pd.DataFrame(us, columns=['$a_{}$'.format(i) for i in range(4)])
actions.plot(subplots=True, layout=(4, 2))

plt.show()

plt.plot(cost_trace)
plt.title('Costo')
plt.show()






