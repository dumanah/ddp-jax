from ddp import DDP
import jax.numpy as jnp
import matplotlib.pyplot as plt
import env
import gym
import numpy as np

if __name__ == '__main__':

    env = gym.make('CartPoleContinuous-v1').env
    n = env.observation_space.shape[0]

    ddp = DDP(lambda x, u: env.next_state(x, u),  # x(i+1) = f(x(i), u)
             lambda x, u: 0.5 * jnp.sum(jnp.square(u)),  # l(x, u)
             lambda x: 0.5 * (100*jnp.square(1.0 - jnp.cos(x[2])) + jnp.square(x[1]) + jnp.square(x[3])),  # lf(x)
             n)

    u_seq = [jnp.zeros(1) for _ in range(ddp.pred_time)]
    x_seq = jnp.array([env.reset()])

    for t in range(ddp.pred_time):
        x_seq = jnp.vstack((x_seq,env.next_state(x_seq[-1], u_seq[t])))

    for _ in range(100):
        
        k_seq, kk_seq,dv = ddp.backward(x_seq, u_seq)
        x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq)

        print("DDP Loop finished!")
        print(u_seq.T)
        observation = env.reset()

        for t in range(ddp.pred_time):
            if t==0:
                print("Simulating with the sequence of control:")
            env.render()
            action = np.asarray(u_seq[t])
            observation, reward, done, info = env.step(action)
    env.close()