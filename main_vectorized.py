from cProfile import run
from time import time
from ddp_vectorized import DDP
import jax.numpy as jnp
from jax import jit
import env
import gym

if __name__ == '__main__':
 
    env = gym.make('CartPoleContinuous-v1').env
    
    next_state = lambda x, u: env.next_states(x, u)  # x(i+1) = f(x(i), u)
    running_cost = lambda x, u: 0.5 * jnp.sum(jnp.square(u)) # l(x, u)
    final_cost = lambda x: 0.5 * (100*jnp.square(1.0 - jnp.cos(x[2])) +  100*jnp.square(1-x[0]) +jnp.square(x[1]) + jnp.square(x[3])) # lf(x)

    dyncst = [next_state,running_cost,final_cost]
    x0 = env.reset()
    pred_time = 50
    u0 = jnp.array(jnp.ones((pred_time,1)))
    ddp = DDP(dyncst, x0, u0)

    '''
    def states_from_policy(x,u):
        for t in range(u.shape[0]):
            x = jnp.vstack((x,next_state(x[-1], u[t])))
            
        return x

    a = jit(states_from_policy)
    tic = time()
    b = a(x_seq,u_seq)
    toc = time() - tic
    all_next_states = ddp.f(x_seq[0],u_seq[0])
    
    
    env.reset()
    for t in range(ddp.pred_time):
            if t==0:
                print("Simulating with the sequence of control:")
            env.render()
            action = jnp.asarray(u_seq[t])
            observation, reward, done, info = env.step(action)
    
    x_seq = [env.reset()]
    
    for t in range(ddp.pred_time):
        x_seq = jnp.vstack((x_seq,next_state(x_seq[-1], u_seq[t])))

    k_seq = [jnp.zeros(1) for _ in range(ddp.pred_time)]
    kk_seq = [jnp.zeros(4) for _ in range(ddp.pred_time)]
    x_seq_i,u_seq_i = ddp.forward(x_seq,u_seq,k_seq,kk_seq,1)
    cost = ddp.l(x_seq_i,u_seq_i) + ddp.lf(x_seq_i[-1])

    
    lmbda = 1
    dlambda = 1
    lambdaFactor = 1.6 
    lambdaMin = 1e-6
    lambdaMax = 1e12
    Alpha = 10**jnp.linspace(0,-3,11)
    
    for _ in range(maxIter):
            
        back_pass_done = False
        tic = time()

        while not back_pass_done:
            k_seq, kk_seq, dv, diverge = ddp.backward(x_seq, u_seq,lmbda)
            
            if diverge:
                dlambda = max(dlambda*lambdaFactor,lambdaFactor)
                lmbda = max(lmbda*dlambda,lambdaMin)
                
                if lmbda > lambdaMax:
                    break
                continue
            back_pass_done = True

        
        fwd_pass_done = False
        if back_pass_done:
            for alpha in Alpha:
                x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq,alpha)
                print(running_cost(x_seq,u_seq) + final_cost(x_seq[-1]))
                cost_new = jnp.sum(jnp.array([ddp.l(x_seq,u_seq),ddp.lf(x_seq[-1])]))
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
                raise ValueError("Failed to find an alpha to find backtracking line-search")
        
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
    
    u_seq = jnp.asarray([ 1.3818403 ,  1.6096525   ,1.493557   , 1.062248   , 0.40467066 ,-0.34089994,
    -1.0235107  ,-1.5066406  ,-1.676355   ,-1.4709579 , -0.9412317  ,-0.24786843,
    0.45664948 , 1.0924271   ,1.5665418  , 1.717066   , 1.424224   , 0.7972192,
    0.11061022 ,-0.5201401  ,-1.1427196  ,-1.6973318  ,-1.920361   ,-1.5663222,
    -0.8251198  ,-0.17325012 , 0.3313924  , 0.91259426 , 1.6144265  , 2.0983377,
    1.8570755  , 0.9613891  , 0.24037194 ,-0.11931868 ,-0.49735647 ,-1.2226162,
    -2.183917   ,-2.6262283 , -1.7867779  ,-0.58791417 ,-0.09437807 ,-0.0413285,
    0.08851875 , 0.78280944 , 2.168895   , 3.2835872  , 2.3968627  , 0.564106,
    -0.22969604, -0.16981263])

    u = [[jnp.array(u_seq[t])] for t in range(u_seq.shape[0])]

    observation = env.reset()
    for t in range(ddp.pred_time):
        if t==0:
            print("Simulating with the sequence of control")
        env.render()
        action = jnp.asarray(u[t])
        observation, reward, done, info = env.step(action)    
    env.close()
    '''