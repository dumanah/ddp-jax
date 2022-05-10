from time import time
from ddp_copy import DDP
import jax.numpy as jnp
from jax import jit, lax
import env
import gym
import warnings
import diff_drive
import timeit

if __name__ == '__main__':
 
    env = gym.make('CartPoleContinuous-v1').env
    
    next_state = lambda x, u: env.next_states(x, u)  # x(i+1) = f(x(i), u)
    running_cost = lambda x, u: 0.5 * jnp.sum(jnp.square(u)) # l(x, u)
    final_cost = lambda x: 0.5 * (10*jnp.square(x[2]) + jnp.square(x[0]) +jnp.square(x[1]) + jnp.square(x[3])) # lf(x)

    dyncst = [next_state,running_cost,final_cost]
    x0 = env.reset()
    pred_time = 10
    u0 = jnp.array(jnp.zeros((pred_time,1)))
    #u0 = [jnp.ones(1)-0.5 if t < pred_time/2 else 0.5-jnp.ones(1) for t in range(pred_time)] #for _ in range(ddp.pred_time)]
    u0 = jnp.asarray(u0)
    ddp = DDP(dyncst, x0, u0)
   
    u_seq = ddp.run_iteration()

    with open('u_seq.npy','wb') as f:
        jnp.save(f,u_seq)

    observation = x0
    env.reset()

    for t in range(pred_time):
        if t==0:
            print("Simulating with the sequence of control")
        env.render()
        action = jnp.asarray(u_seq[t])
        observation, reward, done, info = env.step(action)    
    env.close()
    