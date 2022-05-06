from rlc_continuous import RLC_Continuous
from ddp import DDP
import jax.numpy as jnp
import gym
import matplotlib.pyplot as plt
import numpy as np
import time 

if __name__ == '__main__':

    rlc = RLC_Continuous()
    

    ddp = DDP(lambda x, u: rlc.next_states(x, u),  # x(i+1) = f(x(i), u)
             lambda x, u: 0.5 *0.2* jnp.sum(jnp.square(u)),  # l(x, u)
             lambda x: 0.5 * (100*jnp.square(2 - x[1])) + jnp.square(x[0]),  # lf(x)
             rlc.get_state_dim(),pred_time=50)

    u_seq = [jnp.ones(1) for _ in range(ddp.pred_time)]
    x_seq = jnp.array([rlc.reset()])


    for t in range(ddp.pred_time):
        x_seq = jnp.vstack((x_seq,rlc.next_states(x_seq[-1], u_seq[t])))
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(x_seq[:,0],'b')
    ax1.set_title('i(t)')

    ax2.plot(x_seq[:,1],'r')
    ax2.set_title('V_c(t)')
    fig.suptitle('Step Response of the RLC Circuit, u(t) =1 \n R=1, L=1, C=5')
    plt.show()

    
    for i in range(2):
        start_time = time.time()
        k_seq, kk_seq = ddp.backward(x_seq, u_seq)
        x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq)
        print(f'DDP Iteration {i} took: {time.time()-start_time} seconds')

    x_seq_star = jnp.array([rlc.reset()])
    u_seq_star = u_seq
    for t in range(ddp.pred_time):
        x_seq_star = jnp.vstack((x_seq_star,rlc.next_states(x_seq_star[-1], u_seq_star[t])))
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(x_seq_star[:,0],'b')
    ax1.set_title('i*(t)')

    ax2.plot(x_seq_star[:,1],'r')
    ax2.set_title('V_c*(t)')
    fig.suptitle('Optimal State Trajectories with u*(.)')
    plt.show()
    