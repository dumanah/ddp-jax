from functools import partial
from jax import grad, jacrev, jit, lax, jacfwd, vmap
import jax.numpy as jnp
from time import time
import logging
from tabulate import tabulate
import pyqtgraph as pg
from jax.config import config

config.update("jax_enable_x64", True)


## Switch to using white background and black foreground
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


logging.basicConfig(level=logging.WARNING)


class DDP:
    def __init__(self, dyncst, x0, u0, Op=None):
        next_state, running_cost, final_cost = dyncst
        self.x0 = x0
        self.u0 = u0
        self.backward_pass_done = 0
        self.n = x0.shape[0]  # state dimension
        self.N = u0.shape[0] + 1
        self.regType = 2
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

        # parallel line search
        self.forward_pass_parallel = jit(
            vmap(self.forward_pass, in_axes=[None, None, None, None, 0])
        )

        # Optimization defaults
        """ 
        TODO:
        Can be given to class instance by a dataclass or some other type
        such as Op = ["maxIter": 200, "tolFun": "1e-7"] -> DDP(dyncst,x0,u0,Op)
        https://stackoverflow.com/questions/53376099/python-dataclass-from-a-nested-dict
        """
        if Op == None:
            self.maxIter = 500
            self.mu = 1
            self.dmu = 1
            self.alpha = 1.0
            self.muFactor = 2
            self.muMin = 1e-6
            self.muMax = 1e12
            self.tau = 0.8
            self.tolFun = 1e-7
            self.parallel = True
            self.init_pyqtsubplots(3)

    @partial(jit, static_argnums=(0,))
    def is_pos_def(self, x):
        # return jnp.all(jnp.diagonal(jnp.linalg.cholesky(x)) > 0)
        return jnp.all(jnp.diagonal(jnp.linalg.cholesky(x), axis1=1, axis2=2) > 0)

    @partial(jit, static_argnums=(0,))
    def backward_pass(self, x_seq, u_seq, mu):
        """
        After the observation of high computation time with classic python loop.
        Backward-pass is implemented using jax's while loop, where the condition is checking whether non-positive def. q_uu.
        To understand how it's implemented, see trivial implementation here:
        https://github.com/google/jax/discussions/8375
        """
        V = jnp.zeros(self.N)  # cost-to-go
        V_x = jnp.zeros((self.N, self.n))
        V_xx = jnp.zeros((self.N, self.n, self.n))
        V_xx_reg = jnp.zeros((self.N, self.n, self.n))

        # Initiliazation to be stated from
        V = V.at[-1].set(self.lf(x_seq[-1]))
        V_x = V_x.at[-1].set(self.lf_x(x_seq[-1]))
        V_xx = V_xx.at[-1].set(self.lf_xx(x_seq[-1]))
        k_seq = jnp.zeros((self.N - 1, self.m))
        kk_seq = jnp.zeros((self.N - 1, self.m, self.n))
        dV = jnp.array([0.0, 0.0])

        def backward_loop_cond(carry):
            # these needs to be carried for jax to track
            i, x, u, v_x, v_xx, v_xx_reg, _, _, _ = carry
            f_u_i = self.f_u(x[i], u[i])
            v_xx_reg = v_xx_reg.at[i + 1].set(
                v_xx[i + 1] + mu * jnp.eye(self.n) * (self.regType == 1)
            )
            tmp_reg = f_u_i.T @ v_xx_reg[i + 1]

            q_uu_reg = (
                self.l_uu(x[i], u[i])
                + tmp_reg @ f_u_i
                + mu * (self.regType == 2)
                + jnp.tensordot(v_x[i + 1], jnp.squeeze(self.f_uu(x[i], u[i])), axes=1)
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
                + jnp.tensordot(v_x[i + 1], jnp.squeeze(self.f_xx(x[i], u[i])), axes=1)
            )

            tmp = f_u_i.T @ v_xx[i + 1]

            q_ux = (
                self.l_ux(x[i], u[i])
                + tmp @ f_x_i
                + jnp.tensordot(v_x[i + 1], jnp.squeeze(self.f_ux(x[i], u[i])), axes=1)
            )

            q_uu = (
                self.l_uu(x[i], u[i])
                + tmp @ f_u_i
                + jnp.tensordot(v_x[i + 1], jnp.squeeze(self.f_uu(x[i], u[i])), axes=1)
            )

            v_xx_reg = v_xx_reg.at[i + 1].set(
                v_xx[i + 1] + mu * jnp.eye(self.n) * (self.regType == 1)
            )
            tmp_reg = f_u_i.T @ v_xx_reg[i + 1]

            q_ux_reg = (
                self.l_ux(x[i], u[i])
                + tmp_reg @ f_x_i
                + jnp.tensordot(v_x[i + 1], jnp.squeeze(self.f_ux(x[i], u[i])), axes=1)
            )

            q_uu_reg = (
                self.l_uu(x[i], u[i])
                + tmp_reg @ f_u_i
                + mu * (self.regType == 2)
                + jnp.tensordot(v_x[i + 1], jnp.squeeze(self.f_uu(x[i], u[i])), axes=1)
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
            k = k.at[i].set(k_i)
            kk = kk.at[i].set(kk_i[0])

            return i - 1, x, u, v_x, v_xx, v_xx_reg, k, kk, dv

        seqs_all = lax.while_loop(
            backward_loop_cond,
            backward_loop_body,
            init_val=(self.N - 1, x_seq, u_seq, V_x, V_xx, V_xx_reg, k_seq, kk_seq, dV),
        )

        # if i != 0: divergence occured which why while loop ended
        diverge = seqs_all[0] + 1
        k_seq = seqs_all[6]
        kk_seq = seqs_all[7]
        dv = seqs_all[8]

        return k_seq, kk_seq, dv, diverge

    @partial(jit, static_argnums=(0,))
    def backward_pass_scan(self, x_seq, u_seq, mu):

        # Initiliazation to be stated from
        V_x = self.lf_x(x_seq[-1])
        V_xx = self.lf_xx(x_seq[-1])
        V_xx_reg = jnp.zeros((self.n, self.n))
        k_seq = jnp.zeros(self.m)
        kk_seq = jnp.zeros((self.m, self.n))
        dV = jnp.array([0.0, 0.0])
        Q_uu_reg = jnp.zeros((self.m, self.m))

        def backward_loop_scan(carry, seqs):

            x, u = seqs
            v_x, v_xx, v_xx_reg, k, kk, dv, q_uu_reg = carry
            f_x = self.f_x(x, u)
            f_u = self.f_u(x, u)
            q_x = self.l_x(x, u) + f_x.T @ v_x
            q_u = self.l_u(x, u) + f_u.T @ v_x
            q_xx = (
                self.l_xx(x, u)
                + (f_x.T @ v_xx) @ f_x
                + jnp.tensordot(v_x, jnp.squeeze(self.f_xx(x, u)), axes=1)
            )

            tmp = f_u.T @ v_xx

            q_ux = (
                self.l_ux(x, u)
                + tmp @ f_x
                + jnp.tensordot(v_x, jnp.squeeze(self.f_ux(x, u)), axes=1)
            )

            q_uu = (
                self.l_uu(x, u)
                + tmp @ f_u
                + jnp.tensordot(v_x, jnp.squeeze(self.f_uu(x, u)), axes=1)
            )

            v_xx_reg = v_xx + mu * jnp.eye(self.n) * (self.regType == 1)

            tmp_reg = f_u.T @ v_xx_reg

            q_ux_reg = (
                self.l_ux(x, u)
                + tmp_reg @ f_x
                + jnp.tensordot(v_x, jnp.squeeze(self.f_ux(x, u)), axes=1)
            )

            q_uu_reg = (
                self.l_uu(x, u)
                + tmp_reg @ f_u
                + mu * (self.regType == 2)
                + jnp.tensordot(v_x, jnp.squeeze(self.f_uu(x, u)), axes=1)
            )
            inv_q_uu_reg = jnp.linalg.inv(q_uu_reg)
            k = -inv_q_uu_reg @ q_u
            kk = -inv_q_uu_reg @ q_ux_reg

            dv += jnp.array([k.T @ q_u, 0.5 * (k.T @ q_uu) @ k])

            kk_q_uu = kk.T @ q_uu
            v_x = q_x + kk_q_uu @ k + kk.T @ q_u + q_ux.T @ k
            v_xx = q_xx + kk_q_uu @ kk + kk.T @ q_ux + q_ux.T @ kk

            return (v_x, v_xx, v_xx_reg, k, kk, dv, q_uu_reg), (
                v_x,
                v_xx,
                v_xx_reg,
                k,
                kk,
                dv,
                q_uu_reg,
            )

        _, (_, _, _, k_seq, kk_seq, dv, q_uu_reg) = lax.scan(
            backward_loop_scan,
            (V_x, V_xx, V_xx_reg, k_seq, kk_seq, dV, Q_uu_reg),
            (x_seq[:-1], u_seq),
            reverse=True,
        )

        diverge = ~self.is_pos_def(q_uu_reg)
        return k_seq, kk_seq, dv[0], diverge

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, x_seq, u_seq, k_seq, kk_seq, alpha):
        u0_hat = jnp.zeros(self.m)

        def forward_loop(carry, seqs):
            x_hat, u_hat = carry
            x, u, k, kk = seqs
            control = alpha * k + kk @ (x_hat - x)
            u_hat = u + control
            x_hat = self.f(x_hat, u_hat)
            return (x_hat, u_hat), (x_hat, u_hat)

        _, (x_new, u_new) = lax.scan(
            forward_loop, (self.x0, u0_hat), (x_seq[:-1], u_seq, k_seq, kk_seq)
        )

        x_new = jnp.vstack([self.x0, x_new])
        cost_new = self.l(x_new, u_new) + self.lf(x_new[-1])
        return x_new, u_new, cost_new  # 0: x_seq_hat, 1: u_seq_hat

    def run_iteration(self):
        # initilization of the sequences
        x_seq = jnp.empty((self.N, self.n))
        x_seq = x_seq.at[0].set(self.x0)
        u_seq = self.u0
        k_seq = jnp.zeros((self.N - 1, self.m))
        kk_seq = jnp.zeros((self.N - 1, self.n))

        x_seq, _, cost = self.forward_pass(x_seq, u_seq, k_seq, kk_seq, self.alpha)

        dcost = 0.0  # initialize reduction in cost

        mu = self.mu
        dmu = self.dmu
        muFactor = self.muFactor
        muMin = self.muMin
        muMax = self.muMax
        Alpha = self.tau ** (jnp.linspace(0, 50, 51))

        for i in range(self.maxIter):

            back_pass_done = False
            backward_start = time()

            while not back_pass_done:
                k_seq, kk_seq, dv, diverge = self.backward_pass_scan(
                    x_seq, u_seq, mu
                )

                if diverge:

                    logging.warning(
                        "Encountered non-positive Quu. ---Increasing mu---"
                    )
                    dmu = max(dmu * muFactor, muFactor)
                    mu = max(mu * dmu, muMin)

                    if mu > muMax:
                        break
                else:
                    back_pass_done = True

            # TODO: IMPLEMENTATION OF CHECKING TERMINATION DUE TO SMALL GRADIENT

            backward_finish = time() - backward_start

            fwd_pass_done = False

            if back_pass_done:
                fwd_start = time()
                if self.parallel:
                    X_new, U_new, Cost_new = self.forward_pass_parallel(
                        x_seq, u_seq, k_seq, kk_seq, Alpha
                    )
                    Dcost = cost - Cost_new
                    dcost, w = Dcost.max(axis=0), jnp.argmax(Dcost, axis=0)
                    alpha = Alpha[w]
                    expected = -alpha * (dv[0] + alpha * dv[1])

                    if expected > 0:
                        z = dcost / expected
                    else:
                        z = jnp.sign(dcost)
                        logging.warning(
                            "Non-positive expected reduction of cost: Should not occur!"
                        )
                    if z > 0:
                        fwd_pass_done = True
                        x_new = X_new[w]
                        u_new = U_new[w]
                        cost_new = Cost_new[w]
                else:
                    for alpha in Alpha:
                        x_new, u_new, cost_new = self.forward_pass(
                            x_seq, u_seq, k_seq, kk_seq, alpha
                        )
                        dcost = cost - cost_new
                        expected = -alpha * (dv[0] + alpha * dv[1])

                        if expected > 0:
                            z = dcost / expected
                        else:
                            z = jnp.sign(dcost)
                            logging.warning(
                                "Non-positive expected reduction of cost: Should not occur!"
                            )
                        if z > 0:
                            fwd_pass_done = True
                            break
                        else:
                            logging.warning(
                                "Current alpha: %.5f failed in reducing cost. --- Decreasing alpha ---",
                                alpha,
                            )
                fwd_finish = time() - fwd_start

            if fwd_pass_done:

                # decrease mu
                logging.warning("Backward-pass succeed. ---Decreasing mu---")
                dmu = min(dmu / muFactor, 1 / muFactor)
                mu = mu * dmu * (mu > muMin)

                print(
                    "\n",
                    tabulate(
                        [
                            [
                                i,
                                cost,
                                dcost,
                                expected,
                                fwd_finish,
                                backward_finish,
                            ]
                        ],
                        headers=[
                            "iteration",
                            "cost",
                            "reduction",
                            "expected",
                            "forward-time",
                            "backward-time",
                        ],
                    ),
                )

                self.update_pyqtsubplots(i, cost, mu, alpha)

                # accept changes
                cost = cost_new
                x_seq = x_new
                u_seq = u_new

                if dcost < self.tolFun:
                    print("SUCCESS: cost change < tolFun")
                    break

            else:  # no cost improvement
                logging.warning("No cost improvement, inceasing mu")
                # increase mu
                dmu = max(dmu * muFactor, muFactor)
                mu = max(mu * dmu, muMin)

                # terminate ?
                if mu > muMax:
                    logging.warning("mu reached to it's setted maximum.")
                    break
            if i == self.maxIter:
                logging.warning("Maximum number of Iteration: %d is done ", i)

        return x_seq, u_seq

    def init_pyqtsubplots(self, num_plots):
        self.win = pg.GraphicsLayoutWidget(show=True,size=(1080,360))
        self.win.setWindowTitle("Trajectory Optimization with DDP")
        pg.setConfigOptions(antialias=True)
        self.plts = [self.win.addPlot(row=0,col=i) for i in range(num_plots)]

        # plt.setAutoVisibleOnly(y=True)
        colors = ["b", "r", "k"]
        self.curves = [self.plts[i].plot(pen=pg.mkPen(colors[i],width=2)) for i in range(num_plots)]

        labels = ["Cost", "mu", "Alpha"]
        for i, lbl in enumerate(labels):
            self.plts[i].setLabel("left", lbl)
            self.plts[i].setLabel("bottom", "Iteration")
        self.data = [[] for i in range(num_plots)]

    def update_pyqtsubplots(self, i, V, mu, alpha):

        yy = [V, mu, alpha]
        titles = [
            "Cost",
            "mu (Regularization Param.)",
            "Alpha (Line-Search Param.)",
            "Reduction/Estimated",
        ]
        for k, y in enumerate(yy):
            self.data[k].append(yy[k])
            self.curves[k].setData(jnp.hstack(self.data[k]))
        pg.QtWidgets.QApplication.processEvents()
