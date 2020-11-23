import os
from tqdm import *
import torch
from hessian import hessian
import numpy as np
import scipy
import math

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
        self.GOLD = 0.5*(torch.sqrt(torch.tensor(5.000000)) -1)
    

    
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
            g_k = self.g(x_k)
            print(" Iter:{} x_k : {} y_k: {}".format(j+1, x_k, y))
            if torch.abs(y - x_k) < self.eps or torch.aps(g_k) < self.eps:
                print("Find the optimal!")
                return (x_k, y)
            
            a_0, b_0 = self.init_line_search(x_k=x_k, d_k=d_k, alpha=0.2, gamma=0.1, t=2)
            alpha = self.line_search_alpha(x_k, d_k, a_0, b_0, sigma=1e-5, tao=GOLD)
            w_check = self.strong_wolfe_check(x_k, alpha=alpha, d_k=d_k, theta=0.25, rho=0.1)
            if not w_check:
                print("Stop because not satify wolfe check")
                return (x_k, y)
            else:
                x_k = x_k + alpha*d_k
        
        print("Stop till the max iteration !")
        print("Iter:{}  x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)
    
    def damped_newton_method(self, x_k):
        y_prev = None
        for j in range(self.max_iter):
            y = self.fn(x_k)
            gk = self.g(x_k)
            print("Iter: {}  x_k: {} y_k:{}".format(j+1 , x_k.data, y.data)) 
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            G_k = self.G(x_k)
            d_k = - torch.inverse(G_k).matmul(gk.T).T
            
            a0, b0 = self.init_line_search(x_k=x_k, d_k=d_k, alpha=0.5, gamma=0.1, t = 2)
            alpha = self.line_search_alpha(x_k, d_k, a0, b0, sigma=1e-5, tao=GOLD)
            w_check = self.strong_wolfe_check(x_k, alpha=alpha, d_k=d_k, theta=0.5, rho=0.1)
            # print("alpha:{} d_k:{}".format(alpha, d_k))
            if not w_check:
                print("Stop because violate wolfe check !")
                return(x_k, y)
            else:
                x_k = x_k + alpha*d_k
                y_prev = torch.clone(y)
        
        print("Stop till the max iteration !")
        print("Iter: {} x_k : {} y_k: {}".format(j+1, x_k.data, y.data))
        return (x_k, y)
    
    

    def sr1(self,x_k):
        y_prev = None
        H = torch.eye(len(x_k), dtype=torch.float)
        for j in range(self.max_iter):
            # stop criterion
            y = self.fn(x_k)
            gk = self.g(x_k)
            print("Iter: {}  x_k: {} y_k:{}".format(j+1, x_k.data, y.data))
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            d_k = - H.matmul(gk.T).T

            # print(H)
            a_0, b_0 = self.init_line_search(x_k=x_k, d_k=d_k, alpha=1e-5, gamma=2e-6, t=1.1)
            alpha = self.line_search_alpha(x_k=x_k, d_k=d_k, a_0=a_0, b_0=b_0, sigma=1e-5, tao=GOLD)
            if alpha <= 0 :
                print("Stop because alpha is negative!")
                return (x_k, y)
            # print("alpha:{} d_k:{}".format(alpha, d_k))
            w_check = self.strong_wolfe_check(x_k, alpha=alpha, d_k=d_k, theta=0.3, rho=0.2)
            if not w_check:
                print("Stop because violate wolfe check !")
                return (x_k, y)
            
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

    def bfgs(self, x_k):
        y_prev = None
        H = torch.eye(len(x_k), dtype=torch.float)
        for j in range(self.max_iter):
            # stop criterion
            y = self.fn(x_k)
            gk = self.g(x_k)
            print("Iter: {}  x_k: {} y_k:{}".format(j+1, x_k.data, y.data)) 
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            
            d_k = - H.matmul(gk.T).T
            a_0, b_0 = self.init_line_search(x_k=x_k, d_k=d_k, alpha=1e-5, gamma=1e-6, t=1.1)
            alpha = self.line_search_alpha(x_k=x_k, d_k=d_k, a_0=a_0, b_0=b_0, sigma=1e-5, tao=GOLD)
            # print(alpha)
            if alpha <= 0 :
                print("Stop because alpha is negative!")
                return (x_k, y)
            w_check = self.strong_wolfe_check(x_k, alpha=alpha, d_k=d_k, theta=0.3, rho=0.2)
            if not w_check:
                print("Stop because violate wolfe check !")
                return (x_k, y)
            
            xk_1 = x_k + alpha*d_k
            # yk_1 = self.fn(xk_1)
            gk_1 = self.g(xk_1)
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


    def dfp(self, x_k):
        y_prev = None
        H = torch.eye(len(x_k), dtype=torch.float)
        for j in range(self.max_iter):
            # stop criterion
            y = self.fn(x_k)
            gk = self.g(x_k)
            print("Iter: {}  x_k: {} y_k:{}".format(j+1, x_k.data, y.data)) 
            if torch.norm(gk) < self.eps:
                print("Find optimal x:{} y:{}".format(x_k, y))
                return (x_k, y)
            
            elif j > 0:
                if torch.abs(y - y_prev) < self.eps:
                    print("Find optimal x:{} y:{}".format(x_k.data, y.data))
                    return (x_k, y)
            # print("H : {}".format(H))
            d_k = - H.matmul(gk.T).T
            a_0, b_0 = self.init_line_search(x_k=x_k, d_k=d_k, alpha=1e-5, gamma=1e-6, t=1.3)
            alpha = self.line_search_alpha(x_k=x_k, d_k=d_k, a_0=a_0, b_0=b_0, sigma=1e-5, tao=GOLD)
            if alpha <= 0 :
                print("Stop because alpha is negative!")
                retun(x_k, y)
            w_check = self.strong_wolfe_check(x_k, alpha=alpha, d_k=d_k, theta=0.25, rho=0.1)
            if not w_check:
                print("Stop because not satify wolfe check !")
                return(x_k, y)
            xk_1 = x_k + alpha*d_k
            
            gk_1 = self.g(xk_1)
            sk = xk_1 - x_k
            yk = gk_1 - gk
            # print()
            # print("H : {} sk:{} yk:{} ".format(H.shape, sk.shape, yk.shape))
            sk = sk.view(1,-1)
            yk = yk.view(1,-1)
            H = H + sk.T.matmul(sk)/(sk.matmul(yk.T)) - H.matmul(yk.T).matmul(yk).matmul(H)/(yk.matmul(H).matmul(yk.T))
            # print("H : {}".format(H))
            x_k = xk_1
            y_prev = y
        print("Stop till the max iteration !")
        print(" Iter: {} x_k : {} y_k: {}".format(j+1, x_k, y))
        return (x_k, y)



if __name__ == "__main__":
    print("=============  TEST WOOD FUNCTION =========")
    fn = wood_function
    x0 = torch.tensor([3, -1, 0, 1], dtype=torch.float, requires_grad=True)
    n = 4
    m = 6
    solver = Solver(
        n = n,
        m = m,
        fn = fn,
        x_0 = x0,
        x_minimal = torch.tensor([1,1,1,1], dtype=torch.float),
        y_minimal= 0,
        eps = 1e-8,
        max_iter = 100
    )

    print("\t ## LINE SEARCH ##")

    for j in range(10):
        d_k = torch.tensor([-2, 2, -1, -0], dtype=torch.float)
        # print("y0: {}".format(solver.fn(x0).data))
        # print("dk : {}".format(d_k))
        a0,b0 = solver.init_line_search(x0, d_k, alpha=0.3, gamma=0.06, t=2)
        alpha = solver.line_search_alpha(x0, d_k, a0, b0)
        check  = solver.strong_wolfe_check(x0, alpha=alpha
        ,d_k=d_k, theta=0.2, rho=0.1)
        # print("alpha: ", alpha)
        print("Iter:{0} x_{0}:{1} y_{0}:{2}".format(j, x0.data, solver.fn(x0).data))
        x0 = x0  + alpha*d_k
        # print("xk:", x0.data)
        # print("yk: ", solver.fn(x0).data)
        
        if not check:
            print("Stop because of violating wolfe!")
            print("Find Optimal x:{} y:{}".format((x0 - alpha*d_k).data, solver.fn(x0 - alpha*d_k).data))
            break
    
    print("\t ## DAMPED NEWTON ## ")
    x_k, y = solver.damped_newton_method(solver.x_0)

    print("\t ## SR1 ##")
    x_k, y = solver.sr1(solver.x_0)

    print("\t ## BFGS ##")
    x_k, y = solver.bfgs(solver.x_0)

    print("\t ## DFP ##")
    x_k, y = solver.dfp(solver.x_0)

    m_values = [20, 40, 60, 80, 100]
    for m in m_values:
        print("============= {} =========".format("EXTENDED POWELL SINGULAR FUNCTION (m:{})".format(m)))

        fn = extended_powell_singular_function(m)
        x0 = []
        for i in range(int(m/4)):
            x0 += [3, -1, 0, 1]
        x0 = torch.tensor(x0, dtype=torch.float)
        n = m
        m = m
        solver = Solver(
        n = n,
        m = m,
        fn = fn,
        x_0 = x0,
        x_minimal = torch.tensor([0]*m, dtype=torch.float),
        y_minimal= 0,
        eps = 1e-8,
        max_iter = 100
    )
        print("\t ## LINE SEARCH ##")
        print(fn(solver.x_0))
        for j in range(10):
            d_k = solver.x_minimal - solver.x_0
            
            # print("y0: {}".format(solver.fn(x0).data))
            # print("dk : {}".format(d_k))
            a0,b0 = solver.init_line_search(x0, d_k, alpha=0.3, gamma=0.06, t=2)
            alpha = solver.line_search_alpha(x0, d_k, a0, b0)
            check  = solver.strong_wolfe_check(x0, alpha=alpha
            ,d_k=d_k, theta=0.2, rho=0.1)
            # print("alpha: ", alpha)
            print("Iter:{0} x_{0}:{1} y_{0}:{2}".format(j, x0.data, solver.fn(x0).data))
            x0 = x0  + alpha*d_k
            # print("xk:", x0.data)
            # print("yk: ", solver.fn(x0).data)
            
            if not check:
                print("Stop because of violating wolfe!")
                print("Find Optimal x:{} y:{}".format((x0 - alpha*d_k).data, solver.fn(x0 - alpha*d_k).data))
                break
        
        print("\t ## DAMPED NEWTON ## ")
        x_k, y = solver.damped_newton_method(solver.x_0)

        print("\t ## SR1 ##")
        x_k, y = solver.sr1(solver.x_0)

        print("\t ## BFGS ##")
        x_k, y = solver.bfgs(solver.x_0)

        print("\t ## DFP ##")
        x_k, y = solver.dfp(solver.x_0)
    
    n_values = [20, 40, 60, 80, 100]
    for n in n_values:
        print("============= {} =========".format("TRIGONOMETRIC FUNCTION (n:{})".format(n)))

        fn = trigonometric_function
        
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
        for j in range(10):
            d_k = solver.x_minimal - solver.x_0
            
            # print("y0: {}".format(solver.fn(x0).data))
            # print("dk : {}".format(d_k))
            a0,b0 = solver.init_line_search(x0, d_k, alpha=0.3, gamma=0.06, t=2)
            alpha = solver.line_search_alpha(x0, d_k, a0, b0)
            check  = solver.strong_wolfe_check(x0, alpha=alpha
            ,d_k=d_k, theta=0.2, rho=0.1)
            # print("alpha: ", alpha)
            print("Iter:{0} x_{0}:{1} y_{0}:{2}".format(j, x0.data, solver.fn(x0).data))
            x0 = x0  + alpha*d_k
            # print("xk:", x0.data)
            # print("yk: ", solver.fn(x0).data)
            
            if not check:
                print("Stop because of violating wolfe!")
                print("Find Optimal x:{} y:{}".format((x0 - alpha*d_k).data, solver.fn(x0 - alpha*d_k).data))
                break
        
        print("\t ## DAMPED NEWTON ## ")
        x_k, y = solver.damped_newton_method(solver.x_0)

        print("\t ## SR1 ##")
        x_k, y = solver.sr1(solver.x_0)

        print("\t ## BFGS ##")
        x_k, y = solver.bfgs(solver.x_0)

        print("\t ## DFP ##")
        x_k, y = solver.dfp(solver.x_0)
    



    

