from time import time
from ddp_regularized import DDP
import jax.numpy as jnp
import env
import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    env = gym.make('CartPoleContinuous-v1').env
    
    next_state = lambda x, u: env.next_state(x, u)  # x(i+1) = f(x(i), u)
    running_cost = lambda x, u: 0.5 * jnp.sum(jnp.square(u)) # l(x, u)
    final_cost = lambda x: 0.5 * (500*jnp.square(x[2]) + 500*jnp.square(x[0]) + jnp.square(x[1]) + jnp.square(x[3])) # lf(x)

    dyncst = [next_state,running_cost,final_cost]
    x0 = env.reset()
    pred_time = 20
    u0 = jnp.array(jnp.ones((pred_time-1,1)))
    ddp = DDP(dyncst, x0, u0)
   
    x_seq, u_seq = ddp.run_iteration()

    fig, axs = plt.subplots(1,5)
    axs[0].plot(u_seq[:,0],'b')
    for i in range(5):
        if i == 0:
            axs[i].plot(u_seq[:,0],'b')
        else:
            axs[i].plot(x_seq[:,i-1],'r')
    fig.suptitle('Optimal State Trajectories with u*(.)')
    plt.show(block = True)


    '''

    with open('u_seq.npy','wb') as f:
        jnp.save(f,u_seq)
    '''

    observation = x0
    env.reset()

    for t in range(pred_time):
        if t==0:
            print("Simulating with the sequence of control")
        env.render()
        action = jnp.asarray(u_seq[t])
        observation, reward, done, info = env.step(action)    
    
    env.close()