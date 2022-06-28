from functools import partial
from jax import grad, jacrev, jit, lax, jacfwd
import jax.numpy as jnp
from time import time
import logging
from pygame import K_s
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import itertools

plt.ion()
logging.basicConfig(level=logging.WARNING)


class DDP:
    def __init__(self, dyncst, x0, u0, Op=None):
        next_state, running_cost, final_cost = dyncst
        self.x0 = x0
        self.u0 = u0
        self.backward_pass_done = 0
        self.n = x0.shape[0]  # state dimension
        self.N = u0.shape[0] + 1
        self.regType = 1
        self.m = u0.shape[1]  # control input dimension

        # Step 1: Differentiate the dynamics and the costs
        # jit (just in time compilation) is used to speed up the code
        self.f = jit(next_state)
        self.l = jit(running_cost)
        self.lf = jit(final_cost)
        self.lf_x = jit(grad(self.lf))
        self.lf_xx = jit(jacfwd(self.lf_x))
        self.l_x = jit(grad(self.l, 0))
        self.l_u = jit(grad(self.l, 1))
        self.l_xx = jit(jacfwd(self.l_x, 0))
        self.l_uu = jit(jacfwd(self.l_u, 1))
        self.l_ux = jit(jacfwd(self.l_u, 0))
        self.f_x = jit(jacfwd(self.f, 0))
        self.f_u = jit(jacfwd(self.f, 1))
        self.f_xx = jit(jacfwd(self.f_x, 0))
        self.f_uu = jit(jacfwd(self.f_u, 1))
        self.f_ux = jit(jacfwd(self.f_u, 0))

        # Optimization defaults
        """ 
        TODO:
        Can be given to class instance by a dataclass or some other type
        such as Op = ["maxIter": 200, "tolFun": "1e-7"] -> DDP(dyncst,x0,u0,Op)
        https://stackoverflow.com/questions/53376099/python-dataclass-from-a-nested-dict
        """
        if Op == None:
            self.maxIter = 1000
            self.lmbda = 1
            self.dlambda = 1
            self.alpha = 1
            self.lambdaFactor = 1.05
            self.lambdaMin = 1e-6
            self.lambdaMax = 1e10
            self.tau = 0.95
            self.tolFun = 1e-7

        # plot initialization
        self.fig, self.axs = plt.subplots(2, 2)
        self.axs = self.axs.flatten()
        self.plt = list(itertools.chain(*[ax.plot([], []) for ax in self.axs]))

    @partial(jit, static_argnums=(0,))
    def is_pos_def(self, x):
        return jnp.all(jnp.diagonal(jnp.linalg.cholesky(x)) > 0)

    @partial(jit, static_argnums=(0,))
    def backward_pass(self, x_seq, u_seq, lmbda):

        """
        After the observation of high computation time with classic python loop.
        Backward-pass is implemented using jax's while loop, where the condition is checking whether non-positive def. q_uu.
        To understand how it's implemented, see trivial implementation here:
        https://github.com/google/jax/discussions/8375
        """
        V = jnp.zeros(self.N)  # cost-to-go
        V_x = jnp.array(jnp.zeros((self.N, self.n)))
        V_xx = jnp.array(jnp.zeros((self.N, self.n, self.n)))
        V_xx_reg = jnp.array(jnp.zeros((self.N, self.n, self.n)))

        # Initiliazation to be stated from
        V = V.at[-1].set(self.lf(x_seq[-1]))
        V_x = V_x.at[-1].set(self.lf_x(x_seq[-1]))
        V_xx = V_xx.at[-1].set(self.lf_xx(x_seq[-1]))
        k_seq = jnp.array(jnp.zeros((self.N - 1, self.m)))
        kk_seq = jnp.array(jnp.zeros((self.N - 1, self.n)))
        dV = jnp.array([0.0, 0.0])

        def backward_loop_cond(carry):
            # these needs to be carried for jax to track
            i, x, u, v_x, v_xx, v_xx_reg, _, _, _ = carry
            f_u_i = self.f_u(x[i], u[i])
            v_xx_reg = v_xx_reg.at[i + 1].set(
                v_xx[i + 1] + lmbda * jnp.eye(self.n) * (self.regType == 1)
            )
            tmp_reg = f_u_i.T @ v_xx_reg[i + 1]

            q_uu_reg = (
                self.l_uu(x[i], u[i])
                + tmp_reg @ f_u_i
                + lmbda * (self.regType == 2)
                + jnp.dot(v_x[i + 1], jnp.squeeze(self.f_uu(x[i], u[i])))
            )

            return self.is_pos_def(q_uu_reg) & (i >= 0)

        def backward_loop_body(carry):
            # these needs to be carried for jax to track
            i, x, u, v_x, v_xx, v_xx_reg, k, kk, dv = carry
            f_x_i = self.f_x(x[i], u[i])
            f_u_i = self.f_u(x[i], u[i])
            q_x = self.l_x(x[i], u[i]) + f_x_i.T @ v_x[i + 1]
            q_u = self.l_u(x[i], u[i]) + f_u_i.T @ v_x[i + 1]
            q_xx = (
                self.l_xx(x[i], u[i])
                + (f_x_i.T @ v_xx[i + 1]) @ f_x_i
                + jnp.dot(v_x[i + 1], jnp.squeeze(self.f_xx(x[i], u[i])))
            )

            tmp = f_u_i.T @ v_xx[i + 1]

            q_ux = (
                self.l_ux(x[i], u[i])
                + tmp @ f_x_i
                + jnp.dot(v_x[i + 1], jnp.squeeze(self.f_ux(x[i], u[i])))
            )

            q_uu = (
                self.l_uu(x[i], u[i])
                + tmp @ f_u_i
                + jnp.dot(v_x[i + 1], jnp.squeeze(self.f_uu(x[i], u[i])))
            )

            v_xx_reg = v_xx_reg.at[i + 1].set(
                v_xx[i + 1] + lmbda * jnp.eye(self.n) * (self.regType == 1)
            )
            tmp_reg = f_u_i.T @ v_xx_reg[i + 1]

            q_ux_reg = (
                self.l_ux(x[i], u[i])
                + tmp_reg @ f_x_i
                + jnp.dot(v_x[i + 1], jnp.squeeze(self.f_ux(x[i], u[i])))
            )

            q_uu_reg = (
                self.l_uu(x[i], u[i])
                + tmp_reg @ f_u_i
                + lmbda * (self.regType == 2)
                + jnp.dot(v_x[i + 1], jnp.squeeze(self.f_uu(x[i], u[i])))
            )
            inv_q_uu_reg = jnp.linalg.inv(q_uu_reg)
            k_i = -inv_q_uu_reg @ q_u
            kk_i = -inv_q_uu_reg @ q_ux_reg

            dv += jnp.array([k_i.T @ q_u, 0.5 * (k_i.T @ q_uu) @ k_i])

            kk_i_q_uu = kk_i.T @ q_uu
            v_x = v_x.at[i].set(q_x + kk_i_q_uu @ k_i + kk_i.T @ q_u + q_ux.T @ k_i)
            v_xx = v_xx.at[i].set(
                q_xx + kk_i_q_uu @ kk_i + kk_i.T @ q_ux + q_ux.T @ kk_i
            )
            v_xx = v_xx.at[i].set(0.5 * (v_xx[i].T + v_xx[i]))
            k = k.at[i].set(k_i)
            kk = kk.at[i].set(kk_i.T[0])

            return i - 1, x, u, v_x, v_xx, v_xx_reg, k, kk, dv

        seqs_all = lax.while_loop(
            backward_loop_cond,
            backward_loop_body,
            init_val=(self.N - 1, x_seq, u_seq, V_x, V_xx, V_xx_reg, k_seq, kk_seq, dV),
        )

        # if t != 0: divergence occured which why while loop ended
        diverge = seqs_all[0] + 1
        V_x = seqs_all[3]
        V_xx = seqs_all[4]
        k_seq = seqs_all[6]
        kk_seq = seqs_all[7]
        dv = seqs_all[8]

        return V_x, V_xx, k_seq, kk_seq, dv, diverge

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, x_seq, u_seq, k_seq, kk_seq, alpha):
        x_seq_hat = jnp.array(jnp.zeros(x_seq.shape))
        x_seq_hat = x_seq_hat.at[0].set(self.x0)
        u_seq_hat = jnp.array(jnp.zeros(u_seq.shape))

        def forward_loop(i, seqs):
            x_hat, u_hat, x, u, k, kk = seqs
            control = alpha * k[i] + kk[i] @ (x_hat[i] - x[i])
            u_hat = u_hat.at[i].set(u[i] + control[0])
            x_hat = x_hat.at[i + 1].set(self.f(x_hat[i], u_hat[i]))
            return x_hat, u_hat, x, u, k, kk

        seqs_all = lax.fori_loop(
            0, self.N - 1, forward_loop, (x_seq_hat, u_seq_hat, x_seq, u_seq, k_seq, kk_seq)
        )
        return seqs_all[0], seqs_all[1]  # 0: x_seq_hat, 1: u_seq_hat

    def run_iteration(self):
        # initilization of the sequences
        x_seq = jnp.empty((self.N, self.n))
        x_seq = x_seq.at[0].set(self.x0)
        u_seq = self.u0
        k_seq = jnp.array(jnp.zeros((self.N - 1, self.m)))
        kk_seq = jnp.array(jnp.zeros((self.N - 1, self.n)))

        x_seq, _ = self.forward_pass(x_seq, u_seq, k_seq, kk_seq, 1)

        cost = self.l(x_seq, u_seq) + self.lf(x_seq[-1])  # initial cost
        dcost = 0.0  # initialize reduction in cost

        lmbda = self.lmbda
        dlambda = self.dlambda
        lambdaFactor = self.lambdaFactor
        lambdaMin = self.lambdaMin
        lambdaMax = self.lambdaMax
        Alpha = 0.8 ** (jnp.linspace(0, 50, 51))

        for i in range(self.maxIter):

            back_pass_done = False
            backward_start = time()

            while not back_pass_done:
                V_x, V_xx, k_seq, kk_seq, dv, diverge = self.backward_pass(
                    x_seq, u_seq, lmbda
                )

                if diverge:
                    dlambda = max(dlambda * lambdaFactor, lambdaFactor)
                    lmbda = max(lmbda * dlambda, lambdaMin)

                    if lmbda > lambdaMax:
                        break
                    continue
                back_pass_done = True

            # TODO: IMPLEMENTATION OF CHECKING TERMINATION DUE TO SMALL GRADIENT

            backward_finish = time() - backward_start

            fwd_pass_done = False
            if back_pass_done:
                fwd_start = time()
                for alpha in Alpha:
                    x_new, u_new = self.forward_pass(x_seq, u_seq, k_seq, kk_seq, alpha)
                    running_cost_new, final_cost_new = self.l(x_new, u_new), self.lf(
                        x_new[-1]
                    )
                    cost_new = running_cost_new + final_cost_new
                    dcost = cost - cost_new
                    expected = -alpha * (dv[0] + alpha * dv[1])
                    if expected > 0:
                        z = dcost / expected
                    else:
                        z = jnp.sign(dcost)
                        logging.warn(
                            "Non-positive expected reduction of cost: Should not occur!"
                        )
                    if z > 0:
                        fwd_pass_done = True
                        break
                    else:
                        logging.warn("Current alpha: %.5f failed in reducing cost. -------> Decreasing alpha...",alpha)

                # if not fwd_pass_done:
                #     warnings.warn("Failed to find an alpha to decrease the cost!")

                fwd_finish = time() - fwd_start

            if fwd_pass_done:
                logging.warn("Forward-pass succeed. --------> Decreasing lambda.")
                # decrease lambda
                dlambda = min(dlambda / lambdaFactor, 1 / lambdaFactor)
                lmbda = lmbda * dlambda * (lmbda > lambdaMin)

                # accept changes
                cost = cost_new
                x_seq = x_new
                u_seq = u_new

                self.graphics(i, cost, z, lmbda, alpha)

                print(
                    "\n",
                    tabulate(
                        [
                            [
                                i,
                                cost_new,
                                running_cost_new,
                                final_cost_new,
                                dcost,
                                expected,
                                fwd_finish,
                                backward_finish,
                            ]
                        ],
                        headers=[
                            "iteration",
                            "cost",
                            "running cost",
                            "final cost",
                            "reduction",
                            "expected",
                            "forward-time",
                            "backward-time",
                        ],
                    ),
                )
                if dcost < self.tolFun:
                    print("SUCCESS: cost change < tolFun")
                    break

            else:  # no cost improvement
                logging.warn("No cost improvement, inceasing lambda")
                # increase lambda
                dlambda = max(dlambda * lambdaFactor, lambdaFactor)
                lmbda = max(lmbda * dlambda, lambdaMin)

                # terminate ?
                if lmbda > lambdaMax:
                    logging.warn("lambda reached to it's setted maximum.")
                    break
            if i == self.maxIter:
                logging.warn("Maximum number of Iteration: %d is done ",i)

        return x_seq, u_seq

    def graphics(self, i, V, z, lmbda, alpha):

        yy = [V, lmbda, alpha, z]
        titles = [
            "Cost",
            "Lambda (Regularization Param.)",
            "Alpha (Line-Search Param.)",
            "Reduction/Estimated",
        ]
        colors = ["b", "r", "g", "c"]

        for k, y in enumerate(yy):
            self.plt[k].set_xdata(jnp.append(self.plt[k].get_xdata(), i))
            self.plt[k].set_ydata(jnp.append(self.plt[k].get_ydata(), yy[k]))
            self.plt[k].set_color(colors[k])
            self.axs[k].relim()
            self.axs[k].autoscale_view()
            self.axs[k].set_title(titles[k])

            if y == V:
                self.axs[0].set_yscale("log")

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # Display
        plt.show()
