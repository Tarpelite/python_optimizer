import os
from tqdm import *
import torch
from hessian import hessian
import numpy as np
import scipy

N= 4

def wood_function(x):
    f1 = 10*(x[1] - x[0]**2)
    f2 = 1 - x[0]
    f3 = np.sqrt(90) * (x[3] - x[2]**2)
    f4 = 1 - x[2]
    f5 = np.sqrt(10) * (x[1] + x[3] -2)
    f6 = np.power(10, - 0.5) * (x[1]- x[3])

    return f1**2 + f2**2 + f3**2 + f4**2 + f5**2 + f6**2

def extended_powell_singular_function(x):

    i = 1
    res = 0.000000000
    
    while 4*i <= N:
        f1 = x[4*i-3 -1] + 10 * x[4*i -2 -1]
        f2 = np.sqrt(5) * (x[4*i -1 - 1] - x[4*i -1])
        f3 = (x[4*i -2 -1] - 2*x[4*i -1 -1])**2
        f4 = np.sqrt(10) * (x[4*i -3 -1] - x[4*i -1])** 2
        res += f1**2 + f2**2 + f3**2 + f4**2
        i += 1
    return res

def trigonometric_function(x):
    res = 0.00000000000
    for i in range(len(x)):
        res += len(x) - torch.sum(torch.cos(x)) + (i+1)*(1- torch.cos(x[i])) - torch.sin(x[i])
    return res


class Solver:
    def __init__(self, n=1, m=1, fn=wood_function, x_0=[0], x_minimal=[0], y_minimal=0, eps=10e-8, max_iter=1000):
        '''
        params:
            n: Dimensions
            m: Number of Function definition
            fn: Function definition
            x_0: Standard starting poiny
            x_minimal: the minimal point 
            y_minimal: the minimal value
            eps: the tolerant bias
            max_iter: max iterations
        '''
        self.n = n
        self.m = m
        self.fn = fn
        self.x_0 = x_0
        self.x_minimal = x_minimal
        self.y_minimal = y_minimal
        self.eps = eps
        self.max_iter = max_iter
    

    
    def g(self, x):
        y = self.fn(x)
        y.backward()
        value = torch.clone(x.grad.data)
        x.grad.data.zero_()
        return value
    
    def G(self, x):
        h = hessian(self.fn(x), x, create_graph=True).detach()
        return h
         

    def strong_wolfe_check(self, x_k=[], alpha=1.0, d_k=[],theta=0.25, rho=0.1):
        # Armijo
        lhs = self.fn(x_k + alpha*d_k)
        rhs = self.fn(x_k) + rho*torch.dot(self.g(x_k), d_k)*alpha

        if lhs <= rhs:
            armijo = True
        else:
            armijo = False
        
        # wolfe
        lhs = torch.abs(torch.dot(self.g(x_l + alpha * d_k), d_k))
        rhs = - theta*torch.dot(self.g(x_k), d_k)

        if lsh <= rhs:
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
                    b = min(alpha_, alpha_[i+1])
                    return (a, b)
            elif self.fn(x_k + alpha[i+1]*d_k) > = self.fn(x_k + alpha[i]*d_k):
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

    def line_search_alpha(self, x_k=[], d_k=[], a_0=0, b_0=0, sigma=1e-5, tao=0.618):
        a = [a_0]
        b = [b_0]
        a_l = []
        a_r = []
        i = 0
        for j in self.max_iter:
            if b[i] - a[i] < sigma:
                alpha_star = (b[i] + a [i]) /2
                return alpha_star
            a_l.append( a[i] + (1 - tao)*(b[i] - a[i]))
            a_r.append( a[i] + tao*(b[i] - a[i]) )

            if self.fn(x_k + a[i]_l * d_k) < self.fn(x_k + alpha[i]_r * d_k):
                a.append(a[i])
                b.append(a_r[i])
            else:
                a.append(a_l[i])
                b.append(b[i])
                i = i + 1


            

    def line_search(self, x_k=[], d_k =[], theta=0.25, rho=0.1):
        '''
        params:
            theta: the theta controls the strong wolf principle
            rho: the rho controls the the Armijo principle where 1 > theta > rho > 0
        return:
            (x, y)
            x: the location of the minimize point
            y： the minimal value
        '''

        x_k = x_k
        
        for j in range(self.max_iter):
            y = self.fn(x_k)
            print(" Iter: x_k : {} y_k: {}".format(j+1, x_k, y))
            if torch.abs(y - x_k) < self.eps or torch.aps(g_k) < self.eps:
                print("Find the optimal!")
                return (x_k, y)
            
            a_0, b_0 = self.init_line_search(x_k=x_k, d_k=d_k, alpha=0.2, gamma=0.1, t=2)
            alpha = self.line_search_alpha(x_k, d_k, a_0, b_0, sigma=1e-5, tao=0.618)
            w_check = self.strong_wolfe_check(x_k, alpha=1.0, d_k=d_k, theta=0.25, rho=0.1)
            if not w_check:
                print("Stop because not satify wolfe check")
                return (x_k, y)
            else:
                x_k = x_k + alpha*d_k
        
        print("Stop till the max iteration !")
        print(" Iter: x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)
    
    def damped_newton_method(self, x_k):
        for j in range(self.max_iter):
            y = self.fn(x_k)
            gk = self.g(x_k)
            rint("Iter: {}  x_k: {} y_k:{}".format(j, x_k, y)) 
            if torch.abs(y - x_k) < self.eps or torch.abs(gk) < self.eps:
                print("Find the optimal")
                return (x_k, y)
            
            G_k = self.G(x)
            d_k = - g_k  * torch.inverse(G_k)
            a0, b0 = self.init_line_search(x_k=x_k, d_k=d_k, alpha=0.2, gamma=0.1, t = 2)
            alpha = self.line_search_alpha(x_k, d_k, a_0, b_0, sigma=1e-5, tao=0.618)
            if not w_check:
                print("Stop because not satify wolfe check !")
            else:
                x_k = x_k + alpha*d_k
        
        print("Stop till the max iteration !")
        print(" Iter: x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)
    
    

    def sr1(self,x_k):
        H_0 = torch.eye(len(x_k), dtype=torch.float)
        for j in range(self.max_iter):
            
        

    def bfgs(self):
        pass

    def dfp(self):
        pass

if __name__ == "__main__":
    print("========  TEST WOOD FUNCTION =========")
    solver = Solver()