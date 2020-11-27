import os
from tqdm import *
import torch
from hessian import hessian
import numpy as np
import scipy
import math
from tqdm import trange, tqdm

GOLD = (math.sqrt(5) - 1)*0.50000

def wood_function(x):
    f1 = 10*(x[1] - x[0]**2)
    f2 = 1 - x[0]
    f3 = np.sqrt(90) * (x[3] - x[2]**2)
    f4 = 1 - x[2]
    f5 = np.sqrt(10) * (x[1] + x[3] -2)
    f6 = np.power(10, - 0.5) * (x[1]- x[3])

    return f1**2 + f2**2 + f3**2 + f4**2 + f5**2 + f6**2

def extended_powell_singular_function(n):
    def func(x):
        i = 1
        res = 0.000000000
        while 4*i <= n:
            f1 = x[4*i-3 -1] + 10 * x[4*i -2 -1]
            f2 = np.sqrt(5) * (x[4*i -1 - 1] - x[4*i -1])
            f3 = (x[4*i -2 -1] - 2*x[4*i -1 -1])**2
            f4 = np.sqrt(10) * (x[4*i -3 -1] - x[4*i -1])** 2
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
    

    
    def g(self, x):
        x = x.detach()
        x.requires_grad = True
        y = self.fn(x)
        y.backward()
        value = torch.clone(x.grad.data)
        x.grad.data.zero_()
        x.requires_grad = False
        return value
    
    def G(self, x):
        x.requires_grad = True
        h = hessian(self.fn(x), x, create_graph=True).detach()
        x.requires_grad = False
        return h
         

    def strong_wolfe_check(self, x_k=[], alpha=1.0, d_k=[],theta=0.25, rho=0.1):
        # Armijo
        lhs = self.fn(x_k + alpha*d_k)
        
        rhs = self.fn(x_k) + rho*torch.matmul(self.g(x_k).view(1, -1),d_k.view(-1,1))*alpha
        # print(lhs)
        # print(rhs)
        if lhs <= rhs:
            armijo = True
        else:
            armijo = False
        
        # wolfe
        lhs = torch.matmul(self.g(x_k + alpha * d_k).view(1, -1) , d_k.view(-1, 1))
        rhs =  theta*self.g(x_k).view(1, -1).matmul(d_k.view(-1, 1))
        # print(lhs)
        # print(rhs)
        if lhs >= rhs:
            wolfe = True
        else:
            wolfe = False
        
        return (armijo and wolfe)
    
    def init_line_search(self, x_k=[], d_k=0.1, alpha=0.1, gamma=0.1, t=2):
        i = 0
        alpha = [alpha]
        gamma = [gamma]
        for j in range(self.max_iter):
            alpha.append(alpha[i] + gamma[i])
            if alpha[i+1] <= 0:
                alpha[i+1] = 0
                if i == 0:
                    gamma[i] = - gamma[i]
                    alpha_ = alpha[i+1]
                    continue
                else:
                    a = min(alpha_, alpha[i+1])
                    b = min(alpha_, alpha[i+1])
                    return (a, b)
            elif self.fn(x_k + alpha[i+1]*d_k) >= self.fn(x_k + alpha[i]*d_k):
                if i == 0:
                    gamma[i] = - gamma[i]
                    alpha_ = alpha[i+1]
                    continue
                else:
                    a = min(alpha_, alpha[i+1])
                    b = min(alpha_, alpha[i+1])
                    return (a, b)
            else:
                gamma.append(t*gamma[i])
                alpha_ = alpha[i]
                alpha[i] = alpha[i+1]
                i = i + 1
        a = min(alpha_, alpha[j])
        b = min(alpha_, alpha[j])
        return (a, b)

    def line_search_alpha(self, x_k=[], d_k=[], a_0=0, b_0=0, sigma=1e-5, tao=GOLD):
        a = [a_0]
        b = [b_0]
        a_l = []
        a_r = []
        i = 0
        for j in range(self.max_iter):
            if b[i] - a[i] < sigma:
                alpha_star = (b[i] + a [i]) /2
                return alpha_star
            a_l.append( a[i] + (1 - tao)*(b[i] - a[i]))
            a_r.append( a[i] + tao*(b[i] - a[i]) )

            if self.fn(x_k + a_l[i] * d_k) < self.fn(x_k + a_r[i] * d_k):
                a.append(a[i])
                b.append(a_r[i])
            else:
                a.append(a_l[i])
                b.append(b[i])
                i = i + 1

        alpha_star = (b[i] + a [i]) /2
        return alpha_star
    
    def my_line_search_alpha(self, x_k=0, d_k=0, a_0=0.001, b_0=0.001, eps=1e-5, max_iter=None):
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

    def my_line_search_step(self, x0=0, d0=0, alpha_0=0, gamma_0=0.0001, t=1.2, sigma=0.25, rho=0.1):

        a0, b0 = self.my_init_search_area(x0, d0, alpha_0= alpha_0, gamma_0=gamma_0, t=t)
        for i in range(10):
            alpha = self.my_line_search_alpha(x0, d0, a0, b0)
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
            theta: the theta controls the strong wolf principle
            rho: the rho controls the the Armijo principle where 1 > theta > rho > 0
        return:
            (x, y)
            x: the location of the minimize point
            y： the minimal value
        '''

        
        iter_bar = trange(max_iter) if max_iter is not None else trange(self.max_iter) 
        for j in iter_bar:

            d_k = - self.g(x=x_k)
            alpha = self.my_line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
            y  = self.fn(x_k).data
            
            x_k = x_k + alpha*d_k
            iter_bar.set_description("y:{} alpha:{}".format(y, alpha))
            if torch.abs(self.fn(x_k + alpha*d_k) - y) < self.eps or torch.norm(self.g(x_k)) < self.eps:
                print("Find optimal:")
                print("Iter:{}  x_k : {} y_k: {}".format(j+1, x_k, y))
                return (x_k, y)

            
        print("Stop till the max iteration !")
        print("Iter:{}  x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)
    
    def my_init_search_area(self, x_k=0, d_k=0, alpha_0=0, gamma_0=0.01, t=1.2):
        Alpha = [0.00000]*(self.max_iter + 1)
        Gamma = [0.00000]*(self.max_iter + 1)
        Alpha[0] = alpha_0
        Gamma[0] = gamma_0

        # when i= 0
        alpha = alpha_0
        Alpha[1] = Alpha[0] + Gamma[0]
        for j in range(1, self.max_iter):
            
            if Alpha[1] > 0 and self.fn(x_k + Alpha[1]*d_k) >= self.fn(x_k + Alpha[0]*d_k):
                break
            if Alpha[1] <= 0:
                Alpha[1] = 0
            Gamma[0] = - Gamma[0]
            alpha = Alpha[1]
            Alpha[1] = Alpha[0] + Gamma[0]
            # print(Alpha[1])

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

    def damped_newton_method(self, x_k, alpha_0=0.001, gamma_0=0.001, t=1.2, sigma=0.25, rho=0.1, max_iter=None):
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
            alpha = solver.my_line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
           
            iter_bar.set_description("y:%.8f alpha:%.8f"%(y.data, alpha))
            x_k = x_k + alpha*d_k
            print("alpha:{}".format(alpha))
            y_prev = torch.clone(y)
        
        print("Stop till the max iteration !")
        print("Iter: {} x_k : {} y_k: {}".format(j+1, x_k.data, y.data))
        return (x_k, y)
    
    

    def sr1(self,x_k, alpha_0=0.001, gamma_0=0.001, t=1.2):
        y_prev = None
        H = torch.eye(len(x_k), dtype=torch.float)
        iter_bar = trange(self.max_iter)
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

            alpha = solver.my_line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=0.25, rho=0.1)
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
        y_prev = None
        iter_bar = trange(max_iter) if max_iter is not None else trange(self.max_iter)
        H = torch.eye(len(x_k), dtype=torch.float)
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
            # print(H)
            d_k = - H.matmul(gk.T).T
            alpha = solver.my_line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
            # print("alpha:{} d_k:{}".format(alpha, d_k))
            iter_bar.set_description("y:%.8f alpha:%.8f"%(y, alpha))
            xk_1 = x_k + alpha*d_k
            # yk_1 = self.fn(xk_1)
            gk_1 = self.g(xk_1)
            fk_1 = self.fn(x_k + alpha*d_k)
            
            if fk_1 > y + rho*gk.view(1, -1).matmul(d_k.view(-1, 1))*alpha:
                print("Stop because Armijo !")
                return (x_k, y)
            elif gk_1.view(1, -1).matmul(d_k.view(-1, 1)) < sigma*gk.view(1, -1).matmul(d_k.view(-1, 1)):
                print("Stop because Wolfe !")
                return (x_k, y)
            sk = xk_1 - x_k
            yk = gk_1 - gk

            sk = sk.view(-1, 1)
            yk = yk.view(-1, 1)

            H = H + (torch.eye(sk.size(0)) + yk.T.matmul(H).matmul(yk)/(yk.T.matmul(sk))).matmul((sk.matmul(sk.T))/(yk.T.matmul(sk))) - (sk.matmul(yk.T).matmul(H) + H.matmul(yk).matmul(sk.T))/(yk.T.matmul(sk))
            x_k = xk_1
            y_prev = y
            # print(H)
        print("Stop till the max iteration !")
        print("Iter: {} x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)


    def dfp(self, x_k, alpha_0=0.001, gamma_0=0.001, t=1.2, sigma=0.25, rho=0.1, max_iter = None):
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
            alpha = solver.my_line_search_step(x0=x_k, d0=d_k, alpha_0=alpha_0, gamma_0=gamma_0, t=t,sigma=sigma, rho=rho)
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
    print("=============  TEST WOOD FUNCTION =========")
    fn = wood_function
    x0 = torch.tensor([3, -1, 0, 1], dtype=torch.float)
    solver = Solver(
        fn = fn,
        x_0 = x0,
        x_minimal = torch.tensor([1,1,1,1], dtype=torch.float),
        y_minimal= 0,
        eps = 1e-8,
        max_iter = 100
    )
    print("\t ## LINE SEARCH ##")
    x_ils, y_ils = solver.line_search(x_k=solver.x_0, alpha_0=1e-4, gamma_0=0.01, t=1.2, max_iter=1000)

    print("\t ## DAMPED NEWTON ## ")
    x_dn, y_dn = solver.damped_newton_method(solver.x_0, alpha_0=0.1, gamma_0=0.01, max_iter=1000)

    print("\t ## SR1 ##")
    x_sr1, y_sr1 = solver.sr1(solver.x_0, alpha_0=1e-4, gamma_0=0.01, t=1.2)

    print("\t ## BFGS ##")
    x_bfgs, y_bfgs = solver.bfgs(solver.x_0, alpha_0=1e-4, gamma_0=0.01, t=1.2)

    print("\t ## DFP ##")
    x_dfp, y_dfp = solver.dfp(solver.x_0, alpha_0=1e-4, gamma_0=0.01, t=1.2)

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
        max_iter = 100
    )
        print("\t ## LINE SEARCH ##")
        x_ils, y_ils = solver.line_search(x_k=solver.x_0, alpha_0=1e-5, gamma_0=0.01, t=1.2, max_iter=1000)
   
        
        print("\t ## DAMPED NEWTON ## ")
        x_k, y = solver.damped_newton_method(solver.x_0, alpha_0=0.1, gamma_0=0.01, t=1.2, max_iter=1000)

        print("\t ## SR1 ##")
        x_k, y = solver.sr1(solver.x_0, alpha_0=1e-5, gamma_0=0.01, t=1.2)

        print("\t ## BFGS ##")
        x_k, y = solver.bfgs(solver.x_0, alpha_0=1e-5, gamma_0=0.01, t=1.2, max_iter=100)

        print("\t ## DFP ##")
        x_k, y = solver.dfp(solver.x_0, alpha_0=1e-5, gamma_0=0.01, t=1.2)

def test_trigonometric_function(n_values=[20, 40, 60, 80, 100]):
    for n in n_values:
        print("============= {} =========".format("TRIGONOMETRIC FUNCTION (n:{})".format(n)))

        fn = trigonometric_function(n)
        
        x0 = torch.tensor([1/n]*n, dtype=torch.float)
        
        solver = Solver(
        n = n,
        m = n,
        fn = fn,
        x_0 = x0,
        x_minimal = torch.tensor([0]*n, dtype=torch.float),
        y_minimal= 0,
        eps = 1e-8,
        max_iter = 100
    )
        print("\t ## LINE SEARCH ##")
        print(fn(solver.x_0))
        x0 = solver.x_0
        for j in trange(1000):
            d_k = - solver.g(x=x0)
            
            # print("y0: {}".format(solver.fn(x0).data))

            alpha = solver.my_line_search_step(x0=x0, d0=d_k, alpha_0=0.5, gamma_0=0.001, t=1.2,sigma=0.25, rho=0.1)
            # a0,b0 = solver.my_init_search_area(x0, d_k, alpha_0=0.01, gamma_0=0.001, t=1.2)
            # print("a0:{}, b0:{}".format(a0, b0))
            # alpha = solver.my_line_search_alpha(x0, d_k, a0, b0)
            # check  = solver.strong_wolfe_check(x0, alpha=alpha
            # ,d_k=d_k, theta=0.2, rho=0.1)
            print("alpha: ", alpha)
            print("Iter:{0} x_{0}:{1} y_{0}:{2}".format(j, x0.data, solver.fn(x0).data))
            x0 = x0  + alpha*d_k

            if torch.abs(solver.fn(x0 + alpha*d_k) - solver.fn(x0)) < solver.eps or torch.norm(solver.g(x0)) < solver.eps:
                print("Find optimal:")
                print(x0.data)
                print(solver.fn(x0+alpha*d_k).data)
                break
            
        
        print("\t ## DAMPED NEWTON ## ")
        x_k, y = solver.damped_newton_method(solver.x_0, alpha_0=0.5, gamma_0=0.001)

        print("\t ## SR1 ##")
        x_k, y = solver.sr1(solver.x_0, alpha_0=0.5, gamma_0=0.001)

        print("\t ## BFGS ##")
        x_k, y = solver.bfgs(solver.x_0, alpha_0=0.5, gamma_0=0.001)

        print("\t ## DFP ##")
        x_k, y = solver.dfp(solver.x_0, alpha_0=0.5, gamma_0=0.001)


if __name__ == "__main__":
    test_wood_function()
    test_extended_poweel_singular_function()
    test_trigonometric_function()
   
