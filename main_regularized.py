from time import time
from ddp_regularized import DDP
import jax.numpy as jnp
import numpy as np
import env
import gym
import matplotlib.pyplot as plt
from jax import random

key = random.PRNGKey(0)

if __name__ == "__main__":

    env = gym.make("CartPoleContinuous-v1").env

    x0 = env.reset()  # [x,x_dot,theta,theta_dot]
    pred_time = 200
    u0 = random.uniform(key, (pred_time - 1, 1))
    # u0 = jnp.array(0.5 * jnp.ones((pred_time - 1, 1)))

    def next_state(x, u):
        return env.next_state_diffrax(x, u)  # x(i+1) = f(x(i), u)

    # TODO: plot running cost and final cost to see the optimize the penalty
    def running_cost(x, u):
        return 0.5 * jnp.sum(jnp.square(u))  # l(x, u)

    def final_cost(x):
        return 0.5 * (1000 * jnp.square(x[2]) + 1000 * jnp.square(x[0]))  # lf(x)

    dyncst = [next_state, running_cost, final_cost]

    ddp = DDP(dyncst, x0, u0)

    # Start of the Iteration, returns the optimal control sequence and predicted state.
    x_seq, u_seq = ddp.run_iteration()

    fig, axs = plt.subplots(1, 5)
    axs[0].plot(u_seq[:, 0], "b")
    titles = ["u*(.)", "x^", "x_dot^", "theta^", "theta_dot^"]
    colors = ["b", "r", "g", "c", "k"]
    for i in range(5):
        axs[i].set_title(titles[i])
        if i == 0:
            axs[i].plot(u_seq[:, 0], colors[i])

        else:
            axs[i].plot(x_seq[:, i - 1], colors[i])
    fig.suptitle("Optimal State Trajectories with u*(.)")
    plt.show(block=True)

    observation = x0
    env.reset()
    for t in range(pred_time):
        if t == 0:
            print(
                "--- Simulating the Inverted Pendulum with the control sequence found with DDP ---"
            )
        env.render()
        action = u_seq[t]
        observation, reward, done, info = env.step(action)

    # env.close()
