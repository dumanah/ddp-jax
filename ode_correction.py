from time import time
from ddp_copy import DDP
import jax.numpy as jnp
from jax import jit
import env
import gym
import warnings
import diff_drive
import timeit

if __name__ == '__main__':
 
    env = gym.make('CartPoleContinuous-v1').env
    
    next_state = lambda x,t, u: env.next_states_exp(x,t, u)  # x(i+1) = f(x(i), u)

    x0 = env.reset()
    u0 = jnp.asarray(jnp.zeros(1))
    
    print(next_state(x0,1,u0))
