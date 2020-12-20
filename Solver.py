import os
from re import I
from tqdm import *
import torch
from hessian import hessian
import numpy as np
import scipy
import math
from tqdm import trange, tqdm

GOLD = (math.sqrt(5) - 1)*0.50000

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")




def wood_function(x):
    f1 = 10*(x[1] - x[0]**2)
    f2 = 1 - x[0]
    f3 = torch.sqrt(torch.tensor(90.00)) * (x[3] - x[2]**2)
    f4 = 1 - x[2]
    f5 = torch.sqrt(torch.tensor(10.00)) * (x[1] + x[3] -2)
    f6 = torch.pow(torch.tensor(10.00), - 0.5) * (x[1]- x[3])

    return f1**2 + f2**2 + f3**2 + f4**2 + f5**2 + f6**2

def extended_powell_singular_function(n):
    def func(x):
        i = 1
        res = 0.000000000
        while 4*i <= n:
            f1 = x[4*i-3 -1] + 10 * x[4*i -2 -1]
            f2 = torch.sqrt(torch.tensor(5.000)) * (x[4*i -1 - 1] - x[4*i -1])
            f3 = (x[4*i -2 -1] - 2*x[4*i -1 -1])**2
            f4 = torch.sqrt(torch.tensor(10.000)) * (x[4*i -3 -1] - x[4*i -1])** 2
            res += f1**2 + f2**2 + f3**2 + f4**2
            i += 1
        return res
    return func

def trigonometric_function(n):
    def func(x):
        res = 0.00000000000
        for i in range(n):
            f = len(x) - torch.sum(torch.cos(x)) + (i+1)*(1 - torch.cos(x[i])) - torch.sin(x[i])
            res += f**2
        return res
    return func

def penalty_I_function(n):
    alpha = 1e-5
    def func(x):
        res = 0.00000
        for i in range(n):
            f = torch.sqrt(torch.tensor(alpha)) * (x[i] - 1)
            res += f**2
        
        f = torch.sum(x**2) - 1/4
        res += f**2
        return res
    return func

def extended_rosenbrock_function(n):
    def func(x):
        res = 0.00000
        i = 0
        while 2*i < n:
            f1 = 10 * (x[2*i] - x[i-1]**2)
            f2 = 1 - x[2*i-1]
            res += f1**2 + f2**2
        return res
    return func

class Solver:
    def __init__(self, fn=wood_function, x_0=[0], x_minimal=[0], y_minimal=0, eps=10e-8, max_iter=1000):
        '''
        params:
            fn: Function definition
            x_0: Standard starting poiny
            x_minimal: the minimal point 
            y_minimal: the minimal value
            eps: the tolerant bias
            max_iter: max iterations
        returns:
            An object of optimizer
        '''
        self.fn = fn
        self.x_0 = x_0
        self.x_minimal = x_minimal
        self.y_minimal = y_minimal
        self.eps = eps
        self.max_iter = max_iter
        self.call_cnt = 0
    
    def reset_call_cnt(self):
        '''
        reset function call counter to zero
        '''
        self.call_cnt = 0

    def g(self, x):
        '''
        caculate the derivatives of x
        params:
            x:  float type tensor
        returns:
            g(x) : the same shape tensor as x
        '''
        self.call_cnt += 1
        x = x.detach()
        x.requires_grad = True
        y = self.fn(x)
        y.backward()
        value = torch.clone(x.grad.data)
        x.grad.data.zero_()
        x.requires_grad = False
        return value
    
    def G(self, x):
        '''
        caculate the hessian of x
        params:
            x: float type tensor [1, N]
        returns:
            G(x) : the hessian matrix of x, shape [N , N]
        '''
        self.call_cnt += 1
        x.requires_grad = True
        h = hessian(self.fn(x), x, create_graph=True).detach()
        x.requires_grad = False
        return h
    
    
    def line_search_alpha(self, x_k=0, d_k=0, a_0=0.001, b_0=0.001, eps=1e-5, max_iter=None):
        '''
        GOLD line search
        params:
            x_k : the x value at step k
            d_k : the descent direction at step k
            a_0 : the left boundry of search area
            b_0 : the right boundry of search area
            eps : control the precision of searching
            max_iter: max iterations
        returns:
            alpha: the Optimized step size
        '''
        self.call_cnt += 1
        if max_iter:
            MAX_ITER = max_iter
        else:
            MAX_ITER= self.max_iter
        A = [0.0000] * MAX_ITER
        B = [0.0000] * MAX_ITER
        A[0] = a_0
        B[0] = b_0
        for i in range(MAX_ITER):
            if B[i] - A[i] < eps:
                return (B[i] + A[i])*0.5
            
            Al = A[i] + (1 - GOLD)*(B[i] - A[i])
            Ar = A[i] + GOLD*(B[i]- A[i])

            if self.fn(x_k + Al*d_k) < self.fn(x_k + Ar*d_k):
                A[i+1] = A[i]
                B[i+1] = Ar
            else:
                A[i+1] = Al
                B[i+1] = A[i]
        
        return (A[i] + B[i])*0.5

    def line_search_step(self, x0=0, d0=0, alpha_0=0, gamma_0=0.0001, t=1.2, sigma=0.25, rho=0.1):
        '''
        params:
            x0 : init search point x0
            alpha_0 : init step size
            gamma_0:  init gamma for Gold line search of step size
            t : parameter for Gold line search
            sigma : parameter of wolfe principle
            rho: paramter of wolfe principle
        returns:
            alpha:  optimized step size
        '''
        self.call_cnt += 1

        a0, b0 = self.init_search_area(x0, d0, alpha_0= alpha_0, gamma_0=gamma_0, t=t)
        for i in range(10):
            alpha = self.line_search_alpha(x0, d0, a0, b0)
            fk = self.fn(x0)
            fk_1 = self.fn(x0 + alpha*d0)
            gk_1 = self.g(x0 + alpha*d0)
            gk = self.g(x0)
            # print("({}, {})".format(a0, b0))
            if fk_1 > fk + rho*gk.view(1, -1).matmul(d0.view(-1, 1))*alpha:
                # violate armijo, search a smaller fk_1
                a0_fn = self.fn(x0 + a0*d0)
                b0_fn = self.fn(x0 + b0*d0)
                if a0_fn < b0_fn:
                    b0 = alpha
                else:
                    a0 = alpha

            elif gk_1.view(1, -1).matmul(d0.view(-1, 1)) < sigma*gk.view(1, -1).matmul(d0.view(-1, 1)):
                # violate wolfe, search a bigger gk+1
                a0_fn = self.fn(x0 + a0*d0)
                b0_fn = self.fn(x0 + b0*d0)
                if a0_fn < b0_fn:
                    a0 = alpha
                else:
                    b0 = alpha
            else:
                return alpha
        return alpha

    def line_search(self, x_k=0, alpha_0=0.001, gamma_0=0.001, t=1.2, sigma=0.25, rho=1, max_iter=None):
        '''
        params:
            x_k : init point x_k
            alpha_0 : init step size
            gamma_0:  init gamma for Gold line search of step size
            t : parameter for Gold line search
            sigma : parameter of wolfe principle
            rho: paramter of wolfe principle
            max_iter: the max iteration of the algorithm
        returns:
            x*: Optimal x*
            y: Optimal Value y
        '''
        self.call_cnt += 1
        
        iter_bar = trange(max_iter) if max_iter is not None else trange(self.max_iter) 
        for j in iter_bar:

            d_k = - self.g(x=x_k)
            alpha = self.line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
            y  = self.fn(x_k).data
            
            d_k = d_k / torch.norm(d_k)
            x_k = x_k + alpha*d_k
            iter_bar.set_description("y:{} alpha:{}".format(y, alpha))
            if torch.abs(self.fn(x_k + alpha*d_k) - y) < self.eps or torch.norm(self.g(x_k)) < self.eps:
                print("Find optimal:")
                print("Iter:{}  x_k : {} y_k: {}".format(j+1, x_k, y))
                return (x_k, y)

            
        print("Stop till the max iteration !")
        print("Iter:{}  x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)
    
    def init_search_area(self, x_k=0, d_k=0, alpha_0=0, gamma_0=0.01, t=1.2):
        '''
        params:
            x_k : the x value at step k
            d_k : the descent direction at step k
            alpha_0 : the initial step size
            gamma_0 : to adjuest step size
            t : to adjust step size
        returns:
            a : the left boundry of initial search area
            b : the right boundry of initial search area
        '''
        self.call_cnt += 1
        Alpha = [0.00000]*(self.max_iter + 1)
        Gamma = [0.00000]*(self.max_iter + 1)
        Alpha[0] = alpha_0
        Gamma[0] = gamma_0

        # when i= 0
        alpha = alpha_0
        Alpha[1] = Alpha[0] + Gamma[0]
        # print(d_k.shape)
        if Alpha[1] > 0 and self.fn(x_k + Alpha[1]*d_k) >= self.fn(x_k + Alpha[0]*d_k):
            if Alpha[1] <= 0:
                Alpha[1] = 0
            Gamma[0] = - Gamma[0]
            alpha = Alpha[1]
            Alpha[1] = Alpha[0] + Gamma[0]

        for i in range(1, self.max_iter):
            # step 2
            Alpha[i+1] = Alpha[i] + Gamma[i]
            if Alpha[i+1] <= 0:
                Alpha[i+1] = 0
                # step 4
                a = min(alpha, Alpha[i+1])
                b = max(alpha, Alpha[i+1])
                return (a, b)

            elif self.fn(x_k + Alpha[i+1]*d_k) >= self.fn(x_k + Alpha[i]*d_k):
                # step 4
                a = min(alpha, Alpha[i+1])
                b = max(alpha, Alpha[i+1])
                return (a, b)

            # step 3
            Gamma[i+1] = t*Gamma[i]
            alpha = Alpha[i]
            Alpha[i] = Alpha[i+1] 
        
        a = min(alpha, Alpha[i])
        b = max(alpha, Alpha[i])

        return (a, b)
    
    def LM_method(self, x_k, alpha_0=0.001, gamma_0=0.0001, t=1.2, sigma=0.25, rho=0.1, max_iter=None):
        '''
        params:
            x_k : init point x_k
            alpha_0 : init step size
            gamma_0:  init gamma for Gold line search of step size
            t : parameter for Gold line search
            sigma : parameter of wolfe principle
            rho: paramter of wolfe principle
            max_iter: the max iteration of the algorithm
        returns:
            x*: Optimal x*
            y: Optimal Value y
        '''
        y_prev = None
        MAX_ITER = max_iter if max_iter is not None else self.max_iter
        iter_bar = trange(MAX_ITER)
        vk=0.5
        for j in iter_bar:
            y = self.fn(x_k)
            gk = self.g(x_k)
            # print("Iter: {}  x_k: {} y_k:{}".format(j+1 , x_k.data, y.data)) 
            
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            G_k = self.G(x_k)
            while True:
                G_k = G_k + vk*torch.eye(G_k.size(0))
                try:
                    torch.cholesky(G_k)
                    break
                except Exception as e:
                    continue

            d_k = - torch.inverse(G_k).matmul(gk.T).T
            d_k = d_k / torch.norm(d_k)
            alpha = self.line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
           
            iter_bar.set_description("y:{} alpha:{}".format(y.data, alpha))
            
            x_k = x_k + alpha*d_k
            # print("alpha:{}".format(alpha))
            y_prev = torch.clone(y)
        
        print("Stop till the max iteration !")
        print("Iter: {} x_k : {} y_k: {}".format(j+1, x_k.data, y.data))
        return (x_k, y)

    def damped_newton_method(self, x_k, alpha_0=0.001, gamma_0=0.001, t=1.2, sigma=0.25, rho=0.1, max_iter=None):
        '''
        params:
            x_k : init point x_k
            alpha_0 : init step size
            gamma_0:  init gamma for Gold line search of step size
            t : parameter for Gold line search
            sigma : parameter of wolfe principle
            rho: paramter of wolfe principle
            max_iter: the max iteration of the algorithm
        returns:
            x*: Optimal x*
            y: Optimal Value y
        '''
        y_prev = None
        MAX_ITER = max_iter if max_iter is not None else self.max_iter
        iter_bar = trange(MAX_ITER)
    
        for j in iter_bar:
            y = self.fn(x_k)
            gk = self.g(x_k)
            # print("Iter: {}  x_k: {} y_k:{}".format(j+1 , x_k.data, y.data)) 
            
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            G_k = self.G(x_k)
            d_k = - torch.inverse(G_k).matmul(gk.T).T
            d_k = d_k / torch.norm(d_k)
            alpha = self.line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
           
            iter_bar.set_description("y:{} alpha:{}".format(y.data, alpha))
            
            x_k = x_k + alpha*d_k
            # print("alpha:{}".format(alpha))
            y_prev = torch.clone(y)
        
        print("Stop till the max iteration !")
        print("Iter: {} x_k : {} y_k: {}".format(j+1, x_k.data, y.data))
        return (x_k, y)
    
    def lbfgs(self, x_k, alpha_0=0.001, gamma_0=0.001, t=1.2, sigma=0.25, rho=0.1, max_iter=None, m=8):

        eye = torch.eye(len(x_k), dtype=torch.float)
        H0 = eye

        history_y = []
        history_s = []
        history_rho = []

        iter_bar = trange(max_iter) if max_iter is not None else trange(self.max_iter)

        H = eye
        for j in iter_bar:
            y = self.fn(x_k)
            gk = self.g(x_k)
            # print("Iter: {}  x_k: {} y_k:{}".format(j+1, x_k.data, y.data))
            if torch.norm(gk) < self.eps * max(1, torch.norm(x_k)):
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            if j == 0:
                d_k = - H.matmul(gk.T).T
            d_k = d_k /torch.norm(d_k) 

            alpha = self.line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
            # print("alpha:{} d_k:{}".format(alpha, d_k))
            iter_bar.set_description("y:%.8f alpha:%.8f"%(y, alpha))
            xk_1 = x_k + alpha*d_k
            # yk_1 = self.fn(xk_1)
            gk_1 = self.g(xk_1)
            sk = xk_1 - x_k
            yk = gk_1 - gk
            sk = sk.view(-1, 1)
            yk = yk.view(-1, 1)

            rho_k = 1/(sk.T@yk)
            # print(rho_k.shape)
            # print(yk.shape)
            # print(sk.shape)
            history_y.append(yk)
            history_s.append(sk)
            history_rho.append(rho_k)

            if len(history_y) > m:
                history_y = history_y[1:]
                history_s = history_s[1:]
                history_rho = history_rho[1:]
            
            q = gk.view(-1, 1)
            for i in range(len(history_rho)):
                alpha_i = history_rho[-(i+1)] * history_s[i].T @ q
                q = q - alpha_i * history_y[-(i+1)]
            
            if len(history_rho) > 2:
                gamma_k = history_s[-2].T @ history_y[-2] / (history_y[-2].T @ history_y[-2])
            else:
                gamma_k = history_s[-1].T @ history_y[-1] / (history_y[-1].T @ history_y[-1])
            Hk0 = gamma_k * eye
            z = Hk0 @ q

            for i in range(len(history_rho)):
                beta_i = history_rho[i] * history_y[i].T @ z
                z = z + history_s[i] *(alpha_i - beta_i)
            
            z = -z
            d_k = z.squeeze(1)
            x_k = xk_1
            y_prev = y
            
        print("Stop till the max iteration!")
        print("Iter: {} x_k : {} y_k: {}".format(j+1, x_k.data, y.data))
        return (x_k, y)

    def bb(self, x_k, alpha_0=0.001, gamma_0=0.001, t=1.2, sigma=0.25, rho=0.1, max_iter = None):
        '''
        params:
            x_k : init point x_k
            alpha_0 : init step size
            gamma_0:  init gamma for Gold line search of step size
            t : parameter for Gold line search
            sigma : parameter of wolfe principle
            rho: paramter of wolfe principle
            max_iter: the max iteration of the algorithm
        returns:
            x*: Optimal x*
            y: Optimal Value y
        '''
        y_prev = None
        H = torch.eye(len(x_k), dtype=torch.float)
        iter_bar = trange(max_iter) if max_iter else trange(self.max_iter)
        for j in iter_bar:
            # stop criterion
            y = self.fn(x_k)
            gk = self.g(x_k)
            G = self.G(x_k)
            # print("Iter: {}  x_k: {} y_k:{}".format(j+1, x_k.data, y.data)) 
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            d_k = - gk
            d_k = d_k/ torch.norm(d_k)
            alpha = gk.T @ gk/ (gk.T @ G @ gk)
            iter_bar.set_description("y:{} alpha:{}".format(y, alpha))
           
            xk_1 = x_k + alpha*d_k

            x_k = xk_1
            y_prev = y
        print("Stop till the max iteration !")
        print(" Iter: {} x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)

    def sr1(self,x_k, alpha_0=0.001, gamma_0=0.001, t=1.2, sigma=0.25, rho=0.1, max_iter=None):
        '''
        params:
            x_k : init point x_k
            alpha_0 : init step size
            gamma_0:  init gamma for Gold line search of step size
            t : parameter for Gold line search
            sigma : parameter of wolfe principle
            rho: paramter of wolfe principle
            max_iter: the max iteration of the algorithm
        returns:
            x*: Optimal x*
            y: Optimal Value y
        '''
        y_prev = None
        H = torch.eye(len(x_k), dtype=torch.float)
        iter_bar = trange(max_iter) if max_iter is not None else trange(self.max_iter)
        for j in iter_bar:
            # stop criterion
            y = self.fn(x_k)
            gk = self.g(x_k)
            # print("Iter: {}  x_k: {} y_k:{}".format(j+1, x_k.data, y.data))
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            d_k = - H.matmul(gk.T).T
            d_k = d_k /torch.norm(d_k) 

            alpha = self.line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
            # print("alpha:{} d_k:{}".format(alpha, d_k))
            iter_bar.set_description("y:%.8f alpha:%.8f"%(y, alpha))
            xk_1 = x_k + alpha*d_k
            # yk_1 = self.fn(xk_1)
            gk_1 = self.g(xk_1)
            sk = xk_1 - x_k
            yk = gk_1 - gk
            sk = sk.view(-1, 1)
            yk = yk.view(-1, 1)
            # print("sk: {} yk:{}".format(sk.data, yk.data))
            H = H + (sk - H.matmul(yk)).matmul((sk - H.matmul(yk)).T) / ((sk - H.matmul(yk)).T.matmul(yk))
            x_k = xk_1
            y_prev = y

        print("Stop till the max iteration !")
        print("Iter: {} x_k : {} y_k: {}".format(j+1, x_k.data, y.data))
        return (x_k, y)

    
    def bfgs(self, x_k, alpha_0=0.0001, gamma_0=0.0001, t=1.2, sigma=0.25, rho=0.1, max_iter=None):
        '''
        params:
            x_k : init point x_k
            alpha_0 : init step size
            gamma_0:  init gamma for Gold line search of step size
            t : parameter for Gold line search
            sigma : parameter of wolfe principle
            rho: paramter of wolfe principle
            max_iter: the max iteration of the algorithm
        returns:
            x*: Optimal x*
            y: Optimal Value y
        '''
        y_prev = None
        H = torch.eye(len(x_k), dtype=torch.float)
        iter_bar = trange(max_iter) if max_iter is not None else trange(self.max_iter)
        for j in iter_bar:
            # stop criterion
            y = self.fn(x_k)
            gk = self.g(x_k)
            # print("Iter: {}  x_k: {} y_k:{}".format(j+1, x_k.data, y.data))
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            d_k = - H.matmul(gk.T).T
            d_k = d_k /torch.norm(d_k) 

            alpha = self.line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
            # print("alpha:{} d_k:{}".format(alpha, d_k))
            iter_bar.set_description("y:%.8f alpha:%.8f"%(y, alpha))
            xk_1 = x_k + alpha*d_k
            # yk_1 = self.fn(xk_1)
            gk_1 = self.g(xk_1)
            sk = xk_1 - x_k
            yk = gk_1 - gk
            sk = sk.view(-1, 1)
            yk = yk.view(-1, 1)
            # print("sk: {} yk:{}".format(sk.data, yk.data))
            H = H + (torch.eye(sk.size(0)) + yk.T.matmul(H).matmul(yk)/(yk.T.matmul(sk))).matmul(sk.matmul(sk.T)/(yk.T.matmul(sk))) - ((sk.matmul(yk.T).matmul(H) + H.matmul(yk).matmul(sk.T))/(yk.T.matmul(sk)))
            x_k = xk_1
            y_prev = y

        print("Stop till the max iteration !")
        print("Iter: {} x_k : {} y_k: {}".format(j+1, x_k.data, y.data))
        return (x_k, y)

    def dfp(self, x_k, alpha_0=0.001, gamma_0=0.001, t=1.2, sigma=0.25, rho=0.1, max_iter = None):
        '''
        params:
            x_k : init point x_k
            alpha_0 : init step size
            gamma_0:  init gamma for Gold line search of step size
            t : parameter for Gold line search
            sigma : parameter of wolfe principle
            rho: paramter of wolfe principle
            max_iter: the max iteration of the algorithm
        returns:
            x*: Optimal x*
            y: Optimal Value y
        '''
        y_prev = None
        H = torch.eye(len(x_k), dtype=torch.float)
        iter_bar = trange(max_iter) if max_iter else trange(self.max_iter)
        for j in iter_bar:
            # stop criterion
            y = self.fn(x_k)
            gk = self.g(x_k)
            # print("Iter: {}  x_k: {} y_k:{}".format(j+1, x_k.data, y.data)) 
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            d_k = - H.matmul(gk.T).T
            d_k = d_k/ torch.norm(d_k)
            alpha = self.line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
            iter_bar.set_description("y:{} alpha:{}".format(y, alpha))
           
            xk_1 = x_k + alpha*d_k
            
            gk_1 = self.g(xk_1)
            sk = xk_1 - x_k
            yk = gk_1 - gk
            sk = sk.view(1,-1)
            yk = yk.view(1,-1)
            H = H + sk.T.matmul(sk)/(sk.matmul(yk.T)) - H.matmul(yk.T).matmul(yk).matmul(H)/(yk.matmul(H).matmul(yk.T))
    
            x_k = xk_1
            y_prev = y
        print("Stop till the max iteration !")
        print(" Iter: {} x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)
    
    




def test_wood_function():
    print("=============  TEST WOOD FUNCTION ============")
    fn = wood_function
    # x0 = torch.tensor([-3, -1, -3, -1], dtype=torch.float)
    x0 = torch.tensor([1.01]*4, dtype=torch.float)
    # x1 = torch.tensor([1+1e-7]*4, dtype=torch.float)
    solver = Solver(
        fn = fn,
        x_0 = x0,
        x_minimal = torch.tensor([1,1,1,1], dtype=torch.float),
        y_minimal= 0,
        eps = 1e-8,
        max_iter = 100
    )
    print(solver.g(x0))
    # print(solver.g(x1))
    # print("\t ## LINE SEARCH ##")
    # solver.reset_call_cnt()
    # x_ils, y_ils = solver.line_search(x_k=solver.x_0, alpha_0=0.01, gamma_0=0.0001, t=1.2, max_iter=1500)
    # print("x:{} y:{}".format(x_ils, y_ils))
    # print("call_cnt:{}".format(solver.call_cnt))

    # print("\t ## LM ##")
    # solver.reset_call_cnt()
    # x_lm, y_lm = solver.LM_method(solver.x_0, alpha_0=0.01, gamma_0=0.001, t=1.2, max_iter=1500)
    # print("x:{} y:{}".format(x_lm, y_lm))
    # print("call_cnt:{}".format(solver.call_cnt))

    # print("\t ## DAMPED NEWTON ## ")
    # solver.reset_call_cnt()
    # x_dn, y_dn = solver.damped_newton_method(solver.x_0, alpha_0=0.1, gamma_0=0.001, t=1.2, max_iter=1500)
    # print("x:{} y:{}".format(x_dn, y_dn))
    # print("call_cnt:{}".format(solver.call_cnt))

    # print("\t ## SR1 ##")
    # solver.reset_call_cnt()
    # x_sr1, y_sr1 = solver.sr1(solver.x_0, alpha_0=1e-3, gamma_0=0.001, t=1.2, max_iter=1500)
    # print("x:{} y:{}".format(x_sr1, y_sr1))
    # print("call_cnt:{}".format(solver.call_cnt))

    # print("\t ## BFGS ##")
    # solver.reset_call_cnt()
    # x_bfgs, y_bfgs = solver.bfgs(solver.x_0, alpha_0=0.13, gamma_0=0.01, t=1.2, max_iter=1000)
    # print("x:{} y:{}".format(x_bfgs, y_bfgs))
    # print("call_cnt:{}".format(solver.call_cnt))

    # print("\t ## DFP ##")
    # solver.reset_call_cnt()
    # x_dfp, y_dfp = solver.dfp(solver.x_0, alpha_0=1e-1, gamma_0=0.001, t=1.2, max_iter=1500)
    # print("x:{} y:{}".format(x_dfp, y_dfp))
    # print("call_cnt:{}".format(solver.call_cnt))

    return

def test_extended_poweel_singular_function(m_values=[20, 40, 60, 80, 100]):
    
    for m in m_values:
        print("============= {} =========".format("EXTENDED POWELL SINGULAR FUNCTION (m:{})".format(m)))
       
        fn = extended_powell_singular_function(m)
        x0 = []
        for i in range(int(m/4)):
            x0 += [3, -1, 0, 1]
     
        solver = Solver(
        fn = fn,
        x_0 = torch.tensor(x0, dtype=torch.float),
        x_minimal = torch.tensor([0]*m, dtype=torch.float),
        y_minimal= 0,
        eps = 1e-8,
        max_iter = 100)

        print("\t ## LINE SEARCH ##")
        solver.reset_call_cnt()
        x_ils, y_ils = solver.line_search(x_k=solver.x_0, alpha_0=0.01, gamma_0=0.0001, t=1.2, max_iter=1500)
        print("x:{} y:{}".format(x_ils, y_ils))
        print("call_cnt:{}".format(solver.call_cnt))

        print("\t ## LM ##")
        solver.reset_call_cnt()
        x_lm, y_lm = solver.LM_method(solver.x_0, alpha_0=0.01, gamma_0=0.001, t=1.2, max_iter=1500)
        print("x:{} y:{}".format(x_lm, y_lm))
        print("call_cnt:{}".format(solver.call_cnt))

        print("\t ## DAMPED NEWTON ## ")
        solver.reset_call_cnt()
        x_dn, y_dn = solver.damped_newton_method(solver.x_0, alpha_0=0.1, gamma_0=0.001, t=1.2, max_iter=1500)
        print("x:{} y:{}".format(x_dn, y_dn))
        print("call_cnt:{}".format(solver.call_cnt))

        print("\t ## SR1 ##")
        solver.reset_call_cnt()
        x_sr1, y_sr1 = solver.sr1(solver.x_0, alpha_0=1e-3, gamma_0=0.001, t=1.2, max_iter=1500)
        print("x:{} y:{}".format(x_sr1, y_sr1))
        print("call_cnt:{}".format(solver.call_cnt))

        print("\t ## BFGS ##")
        solver.reset_call_cnt()
        x_bfgs, y_bfgs = solver.bfgs(solver.x_0, alpha_0=1.15, gamma_0=0.001, t=1.5, max_iter=1000)
        print("x:{} y:{}".format(x_bfgs, y_bfgs))
        print("call_cnt:{}".format(solver.call_cnt))

        print("\t ## DFP ##")
        solver.reset_call_cnt()
        x_dfp, y_dfp = solver.dfp(solver.x_0, alpha_0=1e-1, gamma_0=0.001, t=1.2, max_iter=1500)
        print("x:{} y:{}".format(x_dfp, y_dfp))
        print("call_cnt:{}".format(solver.call_cnt))

def test_trigonometric_function(n_values=[1000]):
    for n in n_values:
        print("============= {} =========".format("TRIGONOMETRIC FUNCTION (n:{})".format(n)))

        fn = trigonometric_function(n)
        
        x0 = torch.tensor([1/n]*n, dtype=torch.float).to(device)
        
        solver = Solver(
        fn = fn,
        x_0 = x0,
        x_minimal = torch.tensor([0]*n, dtype=torch.float),
        y_minimal= 0,
        eps = 1e-8,
        max_iter = 100
    )
       
        # print("\t ## LINE SEARCH ##")
        # solver.reset_call_cnt()
        # x_ils, y_ils = solver.line_search(x_k=solver.x_0, alpha_0=0.01, gamma_0=0.0001, t=1.2, max_iter=1500)
        # print("x:{} y:{}".format(x_ils, y_ils))
        # print("call_cnt:{}".format(solver.call_cnt))

        # print("\t ## LM ##")
        # solver.reset_call_cnt()
        # x_lm, y_lm = solver.LM_method(solver.x_0, alpha_0=0.01, gamma_0=0.001, t=1.2, max_iter=1500)
        # print("x:{} y:{}".format(x_lm, y_lm))
        # print("call_cnt:{}".format(solver.call_cnt))

        # print("\t ## DAMPED NEWTON ## ")
        # solver.reset_call_cnt()
        # x_dn, y_dn = solver.damped_newton_method(solver.x_0, alpha_0=0.1, gamma_0=0.001, t=1.2, max_iter=1500)
        # print("x:{} y:{}".format(x_dn, y_dn))
        # print("call_cnt:{}".format(solver.call_cnt))

        # print("\t ## SR1 ##")
        # solver.reset_call_cnt()
        # x_sr1, y_sr1 = solver.sr1(solver.x_0, alpha_0=1e-3, gamma_0=0.001, t=1.2, max_iter=1500)
        # print("x:{} y:{}".format(x_sr1, y_sr1))
        # print("call_cnt:{}".format(solver.call_cnt))

        print("\t ## BFGS ##")
        solver.reset_call_cnt()
        x_bfgs, y_bfgs = solver.bfgs(solver.x_0, alpha_0=1.15, gamma_0=0.001, t=1.5, max_iter=1000)
        print("x:{} y:{}".format(x_bfgs, y_bfgs))
        print("call_cnt:{}".format(solver.call_cnt))

        # print("\t ## LBFGS ##")
        # solver.reset_call_cnt()
        # x_bfgs, y_bfgs = solver.lbfgs(solver.x_0, alpha_0=1.15, gamma_0=0.001, t=1.5, max_iter=1000, m=5)
        # print("x:{} y:{}".format(x_bfgs, y_bfgs))
        # print("call_cnt:{}".format(solver.call_cnt))

        # print("\t ## DFP ##")
        # solver.reset_call_cnt()
        # x_dfp, y_dfp = solver.dfp(solver.x_0, alpha_0=1e-1, gamma_0=0.001, t=1.2, max_iter=1500)
        # print("x:{} y:{}".format(x_dfp, y_dfp))
        # print("call_cnt:{}".format(solver.call_cnt))
        print("\t ## BB ##")
        solver.reset_call_cnt()
        x_bb, y_bb = solver.bb(solver.x_0, alpha_0=1.15, gamma_0=0.001, t=1.5, max_iter=1000)
        print("x:{} y:{}".format(x_bb, y_bb))
        print("call_cnt:{}".format(solver.call_cnt))


def test_penalty_I_function(n_values=[1000]):
    for n in n_values:
        print("============= {} =========".format("PENALTY I FUNCTION (n:{})".format(n)))
        fn = penalty_I_function(n)

        x0 = torch.tensor([i+1 for i in range(n)]).to(device)

        solver = Solver(
            fn = fn,
            x_0 = x0.float(),
            eps=1e-5,
            max_iter=100
        )
        print("\t ## LBFGS ##")
        solver.reset_call_cnt()
        x_lbfgs, y_lbfgs = solver.lbfgs(solver.x_0, alpha_0=1.15, gamma_0=0.001, t=1.5, max_iter=1000)
        print("x:{} y:{}".format(x_lbfgs, y_lbfgs))
        print("call_cnt:{}".format(solver.call_cnt))





if __name__ == "__main__":
    # test_wood_function()
    # test_extended_poweel_singular_function()
    test_penalty_I_function()
    test_trigonometric_function()
