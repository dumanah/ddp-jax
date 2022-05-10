from jax import grad, jacrev, jit, vmap
import jax.numpy as jnp
import time
import warnings

class DDP:
    def __init__(self, next_state, running_cost, final_cost, state_dim, pred_time=50):
        self.pred_time = pred_time
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [jnp.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [jnp.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.v_xx_reg = [jnp.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.backward_pass_done = 0
        self.n = state_dim
        self.regType = 1
        self.control_dim = 1
        #Step 1: Differentiate the dynamics and the costs
        self.f = jit(next_state)
        self.l = jit(vmap(running_cost))
        self.lf = jit(vmap(final_cost))
        self.lf_x = jit(vmap(grad(self.lf)))
        self.lf_xx = jit(vmap(jacrev(self.lf_x)))
        self.l_x = jit(vmap(grad(self.l, 0)))
        self.l_u = jit(vmap(grad(self.l, 1)))
        self.l_xx = jit(vmap(jacrev(self.l_x, 0)))
        self.l_uu = jit(vmap(jacrev(self.l_u, 1)))
        self.l_ux = jit(vmap(jacrev(self.l_u, 0)))
        self.f_x = jit(vmap(jacrev(self.f, 0)))
        self.f_u = jit(vmap(jacrev(self.f, 1)))
        self.f_xx = jit(vmap(jacrev(self.f_x, 0)))
        self.f_uu = jit(vmap(jacrev(self.f_u, 1)))
        self.f_ux = jit(vmap(jacrev(self.f_u, 0)))


    def is_pos_def(self,x):
      return jnp.all(jnp.diagonal(jnp.linalg.cholesky(x)) >0)

    def backward(self, x_seq, u_seq,lmbda):
      self.v[-1] = self.lf(x_seq[-1])
      self.v_x[-1] = self.lf_x(x_seq[-1])
      self.v_xx[-1] = self.lf_xx(x_seq[-1])
      k_seq = []
      kk_seq = []
      dv = jnp.array([0.0,0.0])
      for t in range(self.pred_time - 1, -1, -1):
          f_x_t = self.f_x(x_seq[t], u_seq[t])
          f_u_t = self.f_u(x_seq[t], u_seq[t])
          q_x = self.l_x(x_seq[t], u_seq[t]) + jnp.matmul(f_x_t.T, self.v_x[t + 1])
          q_u = self.l_u(x_seq[t], u_seq[t]) + jnp.matmul(f_u_t.T, self.v_x[t + 1])
          q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
            jnp.matmul(jnp.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
            jnp.dot(self.v_x[t + 1], jnp.squeeze(self.f_xx(x_seq[t], u_seq[t])))
          
          tmp = jnp.matmul(f_u_t.T, self.v_xx[t+1])

          q_ux = self.l_ux(x_seq[t], u_seq[t]) + jnp.matmul(tmp, f_x_t) + \
            jnp.dot(self.v_x[t + 1], jnp.squeeze(self.f_ux(x_seq[t], u_seq[t])))

          q_uu = self.l_uu(x_seq[t], u_seq[t]) + jnp.matmul(tmp, f_u_t) + \
            jnp.dot(self.v_x[t + 1], jnp.squeeze(self.f_uu(x_seq[t], u_seq[t]))) 
          
          
          self.v_xx_reg[t+1] = self.v_xx[t + 1]+ lmbda*jnp.eye(self.n)*(self.regType==1)
          tmp_reg = jnp.matmul(f_u_t.T, self.v_xx_reg[t+1])
          
          q_ux_reg = self.l_ux(x_seq[t], u_seq[t]) + jnp.matmul(tmp_reg, f_x_t) + \
            jnp.dot(self.v_x[t + 1], jnp.squeeze(self.f_ux(x_seq[t], u_seq[t])))
        
          q_uu_reg = self.l_uu(x_seq[t], u_seq[t]) + jnp.matmul(tmp_reg, f_u_t) + lmbda*(self.regType ==2) + \
            jnp.dot(self.v_x[t + 1], jnp.squeeze(self.f_uu(x_seq[t], u_seq[t]))) 
            

          if not self.is_pos_def(q_uu_reg):
            warnings.warn("Q_uu is not positive definite!")
            diverge = True
            break
          else:
            diverge = False

          
          inv_q_uu_reg = jnp.linalg.inv(q_uu_reg)
          k = -jnp.matmul(inv_q_uu_reg, q_u)
          kk = -jnp.matmul(inv_q_uu_reg, q_ux)

          dv += jnp.asarray([jnp.matmul(k.T, q_u), 0.5*jnp.matmul(jnp.matmul(k.T,q_uu),k)])
          #self.v[t] += dv
          kk_t_q_uu = jnp.matmul(kk.T,q_uu)
          self.v_x[t] = q_x + jnp.matmul(kk_t_q_uu,k) + jnp.matmul(kk.T,q_u) + jnp.matmul(q_ux.T,k)
          self.v_xx[t] = q_xx + jnp.matmul(kk_t_q_uu,kk) + jnp.matmul(kk.T,q_ux) + jnp.matmul(q_ux.T,kk)
          self.v_xx[t] = 0.5*(self.v_xx[t].T+self.v_xx[t])
          k_seq.append(k)
          kk_seq.append(kk)
      k_seq.reverse()
      kk_seq.reverse()
      
      return k_seq, kk_seq, dv, diverge

    def forward(self, x_seq, u_seq, k_seq, kk_seq,alpha):
        x_seq_hat = jnp.array(x_seq)
        u_seq_hat = jnp.array(u_seq)
        for t in range(len(u_seq_hat)):
          control = alpha*k_seq[t] + jnp.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
          u_seq_hat = u_seq_hat.at[t].set(u_seq[t] + control[0])
          x_seq_hat = x_seq_hat.at[t + 1].set(self.f(x_seq_hat[t], u_seq_hat[t]))
        
        return x_seq_hat, u_seq_hat
