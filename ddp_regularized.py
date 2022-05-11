from functools import partial
from jax import grad,jacrev,jit, lax
import jax.numpy as jnp
from time import time
import warnings

class DDP:
    def __init__(self, dyncst, x0, u0):

        next_state, running_cost, final_cost = dyncst
        self.x0 = x0
        self.u0 = u0
        self.backward_pass_done = 0
        self.n = x0.shape[0]
        self.N = u0.shape[0] #x0 in place of 
        self.regType = 1
        self.m = u0.shape[1]
      
        #Step 1: Differentiate the dynamics and the costs
        self.f = jit(next_state)
        self.l = jit(running_cost)
        self.lf = jit(final_cost)
        self.lf_x = jit(grad(self.lf))
        self.lf_xx = jit(jacrev(self.lf_x))
        self.l_x = jit(grad(self.l,0))
        self.l_u = jit(grad(self.l, 1))
        self.l_xx = jit(jacrev(self.l_x, 0))
        self.l_uu = jit(jacrev(self.l_u, 1))
        self.l_ux = jit(jacrev(self.l_u, 0))
        self.f_x = jit(jacrev(self.f, 0))
        self.f_u = jit(jacrev(self.f, 1))
        self.f_xx = jit(jacrev(self.f_x, 0))
        self.f_uu = jit(jacrev(self.f_u, 1))
        self.f_ux = jit(jacrev(self.f_u, 0))


        # Optimization defaults
        self.maxIter = 100
        self.lmbda = 1
        self.dlambda = 1
        self.alpha = 1
        self.lambdaFactor = 1.6
        self.lambdaMin = 1e-6
        self.lambdaMax = 1e10

    def is_pos_def(self,x):
      return jnp.all(jnp.diagonal(jnp.linalg.cholesky(x)) > 0)
    
    

    def backward(self, x_seq, u_seq,lmbda):
    
      '''
      
      #v = [0.0 for _ in range(self.N + 1)]
      v = jnp.zeros(self.N+1)
      v_x = jnp.array(jnp.zeros((self.N+1,self.n)))
      v_xx = jnp.array(jnp.zeros((self.N+1,self.n,self.n)))
      v_xx_reg = jnp.array(jnp.zeros((self.N+1,self.n,self.n)))
      v = v.at[-1].set(self.lf(x_seq[-1]))
      v_x = v_x.at[-1].set(self.lf_x(x_seq[-1]))
      v_xx = v_xx.at[-1].set(self.lf_xx(x_seq[-1]))
      k_seq = jnp.array([],dtype=jnp.float32).reshape(0,self.m)
      kk_seq = jnp.array([[]],dtype=jnp.float32).reshape(0,self.n)
      dv = jnp.array([0.0,0.0])

      

      def loop_cond(carry):
        t,(x_seq, u_seq, v_x, v_xx, v_xx_reg, k_seq, kk_seq,dv) = carry
        f_u_t = self.f_u(x_seq[t], u_seq[t])
        v_xx_reg = v_xx_reg.at[t+1].set(v_xx[t + 1]+ lmbda*jnp.eye(self.n)*(self.regType==1))
        tmp_reg = jnp.matmul(f_u_t.T, v_xx_reg[t+1])
        
        q_uu_reg = self.l_uu(x_seq[t], u_seq[t]) + jnp.matmul(tmp_reg, f_u_t) + lmbda*(self.regType ==2) + \
          jnp.dot(v_x[t + 1], jnp.squeeze(self.f_uu(x_seq[t], u_seq[t]))) 
          
        return self.is_pos_def(q_uu_reg) & (t < 1)

      def backward_loop(carry):
        t, (x_seq, u_seq, v_x, v_xx, v_xx_reg, k_seq, kk_seq,dv) = carry
        f_x_t = self.f_x(x_seq[t], u_seq[t])
        f_u_t = self.f_u(x_seq[t], u_seq[t])
        q_x = self.l_x(x_seq[t], u_seq[t]) + jnp.matmul(f_x_t.T, v_x[t + 1])
        q_u = self.l_u(x_seq[t], u_seq[t]) + jnp.matmul(f_u_t.T, v_x[t + 1])
        q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
          jnp.matmul(jnp.matmul(f_x_t.T, v_xx[t + 1]), f_x_t) + \
          jnp.dot(v_x[t + 1], jnp.squeeze(self.f_xx(x_seq[t], u_seq[t])))
        
        tmp = jnp.matmul(f_u_t.T, v_xx[t+1])

        q_ux = self.l_ux(x_seq[t], u_seq[t]) + jnp.matmul(tmp, f_x_t) + \
          jnp.dot(v_x[t + 1], jnp.squeeze(self.f_ux(x_seq[t], u_seq[t])))

        q_uu = self.l_uu(x_seq[t], u_seq[t]) + jnp.matmul(tmp, f_u_t) + \
          jnp.dot(v_x[t + 1], jnp.squeeze(self.f_uu(x_seq[t], u_seq[t]))) 
        
        v_xx_reg = v_xx_reg.at[t+1].set(v_xx[t + 1]+ lmbda*jnp.eye(self.n)*(self.regType==1))
        tmp_reg = jnp.matmul(f_u_t.T, v_xx_reg[t+1])
        
        q_ux_reg = self.l_ux(x_seq[t], u_seq[t]) + jnp.matmul(tmp_reg, f_x_t) + \
          jnp.dot(v_x[t + 1], jnp.squeeze(self.f_ux(x_seq[t], u_seq[t])))
      
        q_uu_reg = self.l_uu(x_seq[t], u_seq[t]) + jnp.matmul(tmp_reg, f_u_t) + lmbda*(self.regType ==2) + \
          jnp.dot(v_x[t + 1], jnp.squeeze(self.f_uu(x_seq[t], u_seq[t]))) 
        
        inv_q_uu_reg = jnp.linalg.inv(q_uu_reg)
        k = -jnp.matmul(inv_q_uu_reg, q_u)
        kk = -jnp.matmul(inv_q_uu_reg, q_ux_reg)

        dv += jnp.asarray([jnp.matmul(k.T, q_u), 0.5*jnp.matmul(jnp.matmul(k.T,q_uu),k)])
        #v[t] += dv
        kk_t_q_uu = jnp.matmul(kk.T,q_uu)
        v_x = v_x.at[t].set(q_x + jnp.matmul(kk_t_q_uu,k) + jnp.matmul(kk.T,q_u) + jnp.matmul(q_ux.T,k))
        v_xx = v_xx.at[t].set(q_xx + jnp.matmul(kk_t_q_uu,kk) + jnp.matmul(kk.T,q_ux) + jnp.matmul(q_ux.T,kk))
        v_xx = v_xx.at[t].set(0.5*(v_xx[t].T+v_xx[t]))
        k_seq = jnp.vstack((k_seq,jnp.asarray(k)))
        kk_seq = jnp.vstack((kk_seq,jnp.asarray(kk)))

        return t+1,(x_seq, u_seq, v_x, v_xx, v_xx_reg, k_seq, kk_seq,dv)

      a = lax.while_loop(loop_cond,backward_loop,init_val=(self.N-1,(x_seq, u_seq, v_x, v_xx, v_xx_reg, k_seq, kk_seq,dv)))[1]

      '''

      for t in range(self.N - 1, -1, -1):
          f_x_t = self.f_x(x_seq[t], u_seq[t])
          f_u_t = self.f_u(x_seq[t], u_seq[t])
          q_x = self.l_x(x_seq[t], u_seq[t]) + jnp.matmul(f_x_t.T, v_x[t + 1])
          q_u = self.l_u(x_seq[t], u_seq[t]) + jnp.matmul(f_u_t.T, v_x[t + 1])
          q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
            jnp.matmul(jnp.matmul(f_x_t.T, v_xx[t + 1]), f_x_t) + \
            jnp.dot(v_x[t + 1], jnp.squeeze(self.f_xx(x_seq[t], u_seq[t])))
          
          tmp = jnp.matmul(f_u_t.T, v_xx[t+1])

          q_ux = self.l_ux(x_seq[t], u_seq[t]) + jnp.matmul(tmp, f_x_t) + \
            jnp.dot(v_x[t + 1], jnp.squeeze(self.f_ux(x_seq[t], u_seq[t])))

          q_uu = self.l_uu(x_seq[t], u_seq[t]) + jnp.matmul(tmp, f_u_t) + \
            jnp.dot(v_x[t + 1], jnp.squeeze(self.f_uu(x_seq[t], u_seq[t]))) 
          
          v_xx_reg = v_xx_reg.at[t+1].set(v_xx[t + 1]+ lmbda*jnp.eye(self.n)*(self.regType==1))
          tmp_reg = jnp.matmul(f_u_t.T, v_xx_reg[t+1])
          
          q_ux_reg = self.l_ux(x_seq[t], u_seq[t]) + jnp.matmul(tmp_reg, f_x_t) + \
            jnp.dot(v_x[t + 1], jnp.squeeze(self.f_ux(x_seq[t], u_seq[t])))
        
          q_uu_reg = self.l_uu(x_seq[t], u_seq[t]) + jnp.matmul(tmp_reg, f_u_t) + lmbda*(self.regType ==2) + \
            jnp.dot(v_x[t + 1], jnp.squeeze(self.f_uu(x_seq[t], u_seq[t]))) 
            
          if not self.is_pos_def(q_uu_reg):
            warnings.warn("Q_uu is not positive definite!")
            diverge = True
            break
          else:
            diverge = False
         
          inv_q_uu_reg = jnp.linalg.inv(q_uu_reg)
          k = -jnp.matmul(inv_q_uu_reg, q_u)
          kk = -jnp.matmul(inv_q_uu_reg, q_ux_reg)

          dv += jnp.asarray([jnp.matmul(k.T, q_u), 0.5*jnp.matmul(jnp.matmul(k.T,q_uu),k)])
          #v[t] += dv
          kk_t_q_uu = jnp.matmul(kk.T,q_uu)
          v_x = v_x.at[t].set(q_x + jnp.matmul(kk_t_q_uu,k) + jnp.matmul(kk.T,q_u) + jnp.matmul(q_ux.T,k))
          v_xx = v_xx.at[t].set(q_xx + jnp.matmul(kk_t_q_uu,kk) + jnp.matmul(kk.T,q_ux) + jnp.matmul(q_ux.T,kk))
          v_xx = v_xx.at[t].set(0.5*(v_xx[t].T+v_xx[t]))
          k_seq = jnp.vstack((k_seq,jnp.asarray(k)))
          kk_seq = jnp.vstack((kk_seq,jnp.asarray(kk)))
      jnp.flip(k_seq)
      jnp.flip(kk_seq)
      
      return k_seq, kk_seq, dv, diverge
    
    def forward(self, x_seq, u_seq, k_seq, kk_seq, alpha):
        x_seq_hat = jnp.array(x_seq)
        u_seq_hat = jnp.array(u_seq)
      
        def f_loop(t,seqs):
          x_seq_hat, u_seq_hat, x_seq, u_seq, k_seq, kk_seq = seqs
          control = alpha*k_seq[t] + jnp.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
          u_seq_hat = u_seq_hat.at[t].set(u_seq[t] + control[0])
          x_seq_hat = x_seq_hat.at[t + 1].set(self.f(x_seq_hat[t], u_seq_hat[t]))
          return x_seq_hat,u_seq_hat,x_seq, u_seq, k_seq, kk_seq

        seqs_all = lax.fori_loop(0,self.N,f_loop,(x_seq_hat,u_seq_hat,x_seq, u_seq, k_seq, kk_seq))
        return seqs_all[0],seqs_all[1] # 0: x_seq_hat, 1: u_seq_hat
    
    def forward_scan(self, x_seq, u_seq, k_seq, kk_seq,alpha):
        x_seq_hat = jnp.array(x_seq)
        u_seq_hat = jnp.array(u_seq)

        for t in range(len(u_seq_hat)):
          control = alpha*k_seq[t] + jnp.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
          u_seq_hat = u_seq_hat.at[t].set(u_seq[t] + control[0])
          x_seq_hat = x_seq_hat.at[t + 1].set(self.f(x_seq_hat[t], u_seq_hat[t]))
        
        return x_seq_hat, u_seq_hat

    def run_iteration(self):

      x_seq = jnp.empty((self.N,self.n))
      x_seq = x_seq.at[0].set(self.x0)
      u_seq = self.u0

      for t in range(self.N):
          x_seq = x_seq.at[t+1].set(self.f(x_seq[t], u_seq[t]))  #use fori_loop

      k_seq = jnp.array(jnp.zeros((self.N,self.m)))
      kk_seq = jnp.array(jnp.zeros((self.N,self.n)))
      x_seq_i,u_seq_i = self.forward(x_seq,u_seq,k_seq,kk_seq,self.alpha) # dont need it 
      cost = self.l(x_seq_i,u_seq_i) + self.lf(x_seq_i[-1])
  
      lmbda = self.lmbda
      dlambda = self.dlambda
      lambdaFactor = self.lambdaFactor 
      lambdaMin = self.lambdaMin
      lambdaMax = self.lambdaMax
      Alpha = 10**jnp.linspace(0,-3,11)
      
      for _ in range(self.maxIter):
              
          back_pass_done = False
          tic = time()

          while not back_pass_done:
              k_seq, kk_seq, dv, diverge = self.backward(x_seq, u_seq,lmbda)
              
              if diverge:
                  dlambda = max(dlambda*lambdaFactor,lambdaFactor)
                  lmbda = max(lmbda*dlambda,lambdaMin)
                  
                  if lmbda > lambdaMax:
                      break
                  continue
              back_pass_done = True

          print("backward pass took: ", time()-tic)
          fwd_pass_done = False
          if back_pass_done:
              fwd_time = time()
              for alpha in Alpha:
                  x_seq, u_seq = self.forward(x_seq, u_seq, k_seq, kk_seq,alpha)
                  print(self.l(x_seq,u_seq) + self.lf(x_seq[-1]))
                  cost_new = jnp.sum(jnp.array([self.l(x_seq,u_seq),self.lf(x_seq[-1])]))
                  dcost = cost - cost_new
                  expected = -alpha*(dv[0] + alpha*dv[1])
                  if(expected > 0):
                      z = dcost / expected
                  else:
                      z = jnp.sign(dcost)
                      warnings.warn(" cost was not decreased!")
                  if(z > 0):
                      fwd_pass_done = True
                      break

              if not fwd_pass_done:
                  warnings.warn("Failed to find an alpha to find backtracking line-search")
                  #fwd_pass_done = True
              print("forward pass took: ", time()-fwd_time)
          if fwd_pass_done:
              dlambda = min(dlambda / lambdaFactor, 1/lambdaFactor)
              lmbda = lmbda * dlambda * (lmbda > lambdaMin)
                  
              cost = cost_new
          else:
              dlambda = max(dlambda*lambdaFactor,lambdaFactor)
              lmbda = max(lmbda*dlambda,lambdaMin)

              if lmbda > lambdaMax:
                  break

          print(f"DDP Loop finished in: {time()-tic}!")
          print(u_seq.T)
          if dcost < 1e-5:
              break
      return u_seq
