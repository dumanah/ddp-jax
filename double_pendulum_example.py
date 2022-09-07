from time import time
from ddp_tensor_vec import DDP
import jax.numpy as jnp
import double_pendulum_env
import gym
import matplotlib.pyplot as plt
from jax import random

key = random.PRNGKey(int(time()))

if __name__ == "__main__":

    env = gym.make("DoubleInvertedPendulum-v1").env

    x0 = env.reset()  # [x, theta, phi, x_dot, theta_dot, phi_dot]
    pred_time = 200
    # u0 = random.uniform(key, (pred_time - 1, 1))
    u0 = jnp.zeros((pred_time - 1, 1))

    def next_state(x, u):
        return env.next_state(x, u)  # x(i+1) = f(x(i), u)

    def running_cost(x, u):
        return 0.1 * (jnp.sum(jnp.square(u)))

    def final_cost(x):  # lf(x)
        return 1000 * (
            jnp.square(1 - x[0]) + jnp.square(x[1]) + jnp.square(x[2])
        )  # x  # theta # phi

    dyncst = [next_state, running_cost, final_cost]

    ddp = DDP(dyncst, x0, u0)

    # Start of the Iteration, returns the optimal control sequence and predicted state.
    start = time()
    x_seq, u_seq = ddp.run_iteration()
    print("finished in", time() - start)

    fig, axs = plt.subplots(1, 7)
    axs[0].plot(u_seq[:, 0], "b")
    titles = ["u*(.)", "x^", "theta", "phi", "x_dot^", "theta_dot^", "phi_dot"]
    colors = ["b", "r", "g", "c", "k", "m", "y"]
    for i in range(7):
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
