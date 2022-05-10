import jax.numpy as jnp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

from jax.experimental.ode import odeint


class CartPoleContinuousEnv():

    def __init__(self):

        self.gravity = 9.8
        self.M = 1.0
        self.m = 0.1
        self.total_m = (self.m + self.M)
        self.l = 0.5  # actually half the pole's length
        self.ml = (self.m * self.l)
        self.mgl = self.ml*self.gravity
        self.mg = self.m*self.gravity
        self.tau = 0.01

        self.state = None

    def _state_eq_jnp(self, st, t, u):
        x, vel, theta, omega = st
        force = u[0]
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        calc_den_help = self.M + (self.m * (sintheta**2))

        vel_dot = (force + self.ml*sintheta*(omega**2) - self.mg * costheta * sintheta) / calc_den_help
        omega_dot = (-force*costheta - self.ml*sintheta*costheta*(omega**2) + self.total_m*self.gravity*sintheta) / (self.l*calc_den_help)

        return jnp.array([vel,vel_dot,omega,omega_dot])
    
    def _state_eq_np(self, t, st, u):
        x, vel, theta, omega = st
        force = u[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        calc_den_help = self.M + (self.m * (sintheta**2))

        vel_dot = (force + self.ml*sintheta*(omega**2) - self.mg * costheta * sintheta) / calc_den_help
        omega_dot = (-force*costheta - self.ml*sintheta*costheta*(omega**2) + self.total_m*self.gravity*sintheta) / (self.l*calc_den_help)

        return np.array([vel,vel_dot,omega,omega_dot])


    def _next_states_jnp(self,st,u):
        sol = odeint(self._state_eq_jnp,st, jnp.linspace(0., self.tau,6),u)
        return jnp.asarray(sol.T[:,-1])
   
    def _next_states_np(self,st,u):
        sol = solve_ivp(self._state_eq_np,y0=st,t_span=[0,self.tau],t_eval=np.linspace(0.,self.tau,6),args=(u,))
        return np.asarray(sol.y[:,-1])

    def _reset(self):
        self.state = jnp.array([0.0, 0.0, jnp.pi, 0.0])
        return jnp.array(self.state)

    def _plot_sim(self, st,u):
        sol = solve_ivp(self._state_eq_np,y0=st,t_span = [0,10],t_eval = np.arange(0, 10, 0.01),args=(u,))
        plt.plot(sol.y[1,:])
        plt.show





