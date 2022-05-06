import jax.numpy as jnp
from jax.experimental.ode import odeint


class RLC_Continuous():

    def __init__(self,R=1,L=1,C=5):
        self.resistance = R
        self.inductance = L
        self.capacity = C
        self.tau = 0.5  # seconds between state updates
        self.state_dim = 2  # (i(t),v_c(t))
        self.state = None

    def _state_eq(self, states, t, u):
        i, v_c = states
        v_i = u[0]

        i_dot = (-self.resistance*i - v_c + v_i) / self.inductance 
        v_c_dot = (i / self.capacity) 
        return jnp.array([i_dot, v_c_dot])

    def next_states(self,st,u):
        sol = odeint(self._state_eq,st, jnp.linspace(0., self.tau,2),u)
        return jnp.asarray(sol.T[:,-1])

    def step(self, action, integration='odeint'):
        
        if integration == 'odeint':
            assert self.state is not None, "Call reset before using step method."
            state = self.state
            self.state = self.next_states(state,action)
            i, v_c = self.state

        return jnp.array(self.state)

    def get_state_dim(self):

        return self.state_dim
    def reset(self):
        self.state = jnp.array([0.0,0.0])
        return jnp.array(self.state)