from diffrax import diffeqsolve, ODETerm, Dopri5, Euler
import jax.numpy as jnp
from jax import jit, grad
import env
import gym
from time import time
env = gym.make('CartPoleContinuous-v1').env
    
next_state = lambda x, u: env.next_states(x, u) 



x0 = jnp.array([0., 0., jnp.pi, 0.])
u0 = jnp.array(jnp.zeros((200,1)))

print(env.next_states_diffrax(x0,u0[0]))
print(env.next_states(x0,u0[0]))

observation = x0

env.reset()

for t in range(200):
    if t==0:
        print("Simulating with the sequence of control")
    env.render()
    action = jnp.asarray(u0[t])
    observation, reward, done, info = env.step(action)    
env.close()
