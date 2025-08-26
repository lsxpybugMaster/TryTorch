from trytorch.autograd import Tensor
from trytorch import nn
from typing import List

class Optimizer:

    def __init__(self, params:List[Tensor]):
        self.params = params

    def step(self):
        raise NotImplementedError()
    
    def reset_grad(self):
        for p in self.params:
            p.grad = None


'''
带正则项及动量机制(EMA)的SGD 
    grad_t  = grad(w_t) + λw_t          带L2正则的梯度
    u_t     = μu_(t-1) + (1 - μ)grad_t  梯度动量累积
    w_(t+1) = w_t - ηu_t                参数更新
'''
class SGD(Optimizer):
    '''
        默认是普通SGD
    '''
    def __init__(
        self, 
        params:List[Tensor], 
        lr=0.01, 
        momentum=0.0,    # 动量系数
        weight_decay=0.0 # 正则项系数
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {} 
        self.weight_decay = weight_decay


    def step(self):
        for param in self.params:
            
            #  grad(w_t) + λw_t 
            grad_t = param.grad.cached_data + self.weight_decay * param.cached_data
            
            #  μu_(t-1) + (1 - μ)grad_t
            u_t = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * grad_t
            if self.momentum:
                self.u[param] = u_t
            
            # w_(t+1) = w_t - ηu_t   
            # SGD 参数更新
            param.cached_data -= self.lr * u_t



class Adam(Optimizer):

    def __init__(
        self,
        params:List[Tensor], 
        lr = 0.01,
        beta1 = 0.9,   # 一阶矩动量权重
        beta2 = 0.999, # 二阶矩动量权重
        eps = 1e-8,
        weight_decay = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # 时间步,用于偏差修正时作为β的次幂

        self.m = {}
        self.v = {}


    def step(self):
        self.t += 1
        for param in self.params:
            
            # 计算带正则的梯度  grad_(w_t) + λw_t
            grad_t = param.grad.cached_data + self.weight_decay * param.cached_data
        
            # 更新m, v
            '''
                mt = β1*m_(t-1) + (1 - β1)gt
                vt = β2*v_(t-1) + (1 - β2)gt**2
            '''
            m_t = self.beta1 * self.m.get(param,0) + (1 - self.beta1) * grad_t
            v_t = self.beta2 * self.v.get(param,0) + (1 - self.beta2) * grad_t ** 2
            
            self.m[param] = m_t
            self.v[param] = v_t

            # 偏差修正
            m_t_hat = m_t / (1 - self.beta1 ** self.t)
            v_t_hat = v_t / (1 - self.beta2 ** self.t)

            # 更新参数
            '''
                w_(t+1) = w_t - η / (v_t ** 0.5 + ε) * m_t
            '''
            param.cached_data -= self.lr / (v_t_hat ** 0.5 + self.eps) * m_t_hat 

            
