from jax import grad, jacobian
import jax.numpy as jnp
from sympy import prime

class DDP:
    def __init__(self, next_state, running_cost, final_cost, state_dim, pred_time=50):
        self.pred_time = pred_time
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [jnp.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [jnp.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.f = next_state
        self.lf = final_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)
        self.l_x = grad(running_cost, 0)
        self.l_u = grad(running_cost, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_uu = jacobian(self.f_u, 1)
        self.f_ux = jacobian(self.f_u, 0)

    def is_pos_def(self,x):
      return jnp.all(jnp.linalg.eigvalsh(x)>0)

    def backward(self, x_seq, u_seq):
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t]) + jnp.matmul(f_x_t.T, self.v_x[t + 1])
            q_u = self.l_u(x_seq[t], u_seq[t]) + jnp.matmul(f_u_t.T, self.v_x[t + 1])
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
              jnp.matmul(jnp.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
              jnp.dot(self.v_x[t + 1], jnp.squeeze(self.f_xx(x_seq[t], u_seq[t])))
            tmp = jnp.matmul(f_u_t.T, self.v_xx[t + 1])
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + jnp.matmul(tmp, f_u_t) + \
              jnp.dot(self.v_x[t + 1], jnp.squeeze(self.f_uu(x_seq[t], u_seq[t])))

            if not self.is_pos_def(q_uu):
              raise ValueError("Q_uu is not positive definite!")
            q_ux = self.l_ux(x_seq[t], u_seq[t]) + jnp.matmul(tmp, f_x_t) + \
              jnp.dot(self.v_x[t + 1], jnp.squeeze(self.f_ux(x_seq[t], u_seq[t])))
            inv_q_uu = jnp.linalg.inv(q_uu)
            k = -jnp.matmul(inv_q_uu, q_u)
            kk = -jnp.matmul(inv_q_uu, q_ux)
            dv = 0.5 * jnp.matmul(q_u, k)
            self.v[t] += dv
            self.v_x[t] = q_x - jnp.matmul(jnp.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[t] = q_xx + jnp.matmul(q_ux.T, kk)
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = jnp.array(x_seq)
        u_seq_hat = jnp.array(u_seq)
        for t in range(len(u_seq)):
            control = k_seq[t] + jnp.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat = u_seq_hat.at[t].set(u_seq[t] + control)
            x_seq_hat = x_seq_hat.at[t + 1].set(self.f(x_seq_hat[t], u_seq_hat[t]))
        return x_seq_hat, u_seq_hat

