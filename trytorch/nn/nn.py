from typing import List
from trytorch import ops, init, array_api
from trytorch.autograd import Tensor
import math

'''
模块部分,提供用于构建模型的模块
层次结构:
    Net(定义完的网络): Module        -> 
    SubModule: Module               ->
    SubModule.self.__dict__: dict   ->
    TensorList : List[Tensor]      
以上层次可能递归多次!
'''

#------------------------------------核心部分-----------------------------

class Module:
    '''
        所有模块的基类
    '''
    def __init__(self):
        ''' 子模块定义内部计算模块 '''
        # 🐍 Python类的self变量会存储在 self.__dict__ 中
        self.training = True


    def forward(self, *args, **kwargs):
        '''子模块实现前向计算函数'''
        raise NotImplementedError()    


    def parameters(self) -> List[Tensor]:
        '''获取模块存储的所有Tensor'''
        return _unpack_params(self.__dict__)


    def _children(self) -> List["Module"]:
        '''辅助函数,将网络分解成小模块'''
        return _child_modules(self.__dict__)


    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False


    def train(self):
        self.training = True
        for m in self._children():
            m.training = True


    def __call__(self, *args, **kwargs):
        '''
            通过网络实例名直接调用前向计算过程
            __call__中不加额外逻辑时(如pytorch中加了钩子函数):
                net(*args,**kwargs) <=> net.forward(*args,**kwargs)
        '''
        return self.forward(*args, **kwargs)


    def print(self):
        '''打印模块的结构'''
        print(self.__dict__)



class Parameter(Tensor):
    '''
        标签类,代表这是需要更新梯度的Tensor
    '''


def _unpack_params(value: object) -> List[Tensor]:
    '''
        将模块中Tensor提取,用于之后操作(传入优化器更新参数)
    '''
    # 是需要更新梯度的Tensor,直接返回
    if isinstance(value, Parameter):
        return [value]
   
    # 是 Module, 则通过self.__dict__解析 递归调用
    elif isinstance(value, Module):
        # value.parameters() :: _unpack_params(self.__dict__)
        return value.parameters()
    
    # 是 dict, 则提取成List
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params

    # Sequential
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    
    # Tensor而非Parameter时,不返回Tensor
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    """
    目前这个函数主要服务于module的train和eval状态的切换
    """
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules

    # FIXME : 参考代码这里为if
    elif isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    
    else:
        return []

#------------------------------------实际模块定义部分-----------------------------

class Identity(Module):
    def forward(self, x):
        return x
    


class Linear(Module):
    
    def __init__(
        self, 
        in_features, 
        out_features,
        bias = True,
        device = None,
        dtype = "float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Parameter即需要更新参数的Tensor
        self.weight = Parameter(
            init.kaiming_uniform(in_features,out_features,device=device,dtype=dtype,requires_grad=True)
        )        

        # b已经自动被转置了,无需再转置
        self.bias = Parameter(
            ops.transpose(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype,requires_grad=True))
        ) if bias else None


    '''
        Y = X @ W + B     b : (1, out_features) => B : (n, out_features)
    '''
    def forward(self, X: Tensor) -> Tensor:
        Y = ops.matmul(X, self.weight)
        if self.bias:
            bias = ops.broadcast_to(self.bias, Y.shape)
            Y += bias
        return Y



class Flatten(Module):
    def forward(self, X: Tensor):
        # (batch,f1,f2,f3) => (batch,f1 * f2 * f3)
        return ops.reshape(X, (X.shape[0],-1))
    


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
    


class Sequential(Module):
    '''
        存储一系列顺序计算的模块,一次前向调用计算完毕
    '''
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x
    


'''
    合并简化(Softmax + 交叉熵损失)
    CrossEntropy = -sum(ylog(p)) ==> -log(q_k)
    z_k 来自 logit, 需经过如下 Softmax:
    q_k = Softmax(z_k) = e**z_k / sum(e**i)
    合并带入:
        Loss = -log(q_k)
             = -log(e**z_k / sum(e**i))
             = -(log(e**z_k)) - logsum(e**i))
             = -(z_k - logsum)
             = logsum - z_k
'''
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        '''
            logits:  (batch, embed)  embed维数与类别相同
        '''
        batch_size = logits.shape[0]
        classes = logits.shape[1]
        # logsum
        normalize_x = ops.logsumexp(logits, axes=1)
        
        y_one_hot = init.one_hot(classes, y, device=y.device)
       
        # 提取出Z_y (即上面的z_k的向量形式,包含所有对应真实标签位置的logit值)
        Z_y = ops.summation(logits * y_one_hot, axes=1)

        loss = ops.summation(normalize_x - Z_y)

        return loss / batch_size


'''
训练用 batch 统计，推理用全局统计
    x_n = (x - μ) / sqrt(σ**2 + ε)
    y = γx_n + β
'''
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1,device=None, dtype='float32'):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # γ, 初始全1
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad = True))
        # β, 初始全0
        self.bias   = Parameter(init.zeros(self.dim,device=device, dtype=dtype, requires_grad = True))
        # 动量系数无需学习
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var  = init.ones(self.dim, device=device, dtype=dtype)

    def forward(self, x):
        # x : (b, features = self.dim)
        batch_size = x.shape[0]

        mean = x.sum((0,)) / batch_size
        var = ((x - mean) ** 2).sum((0,)) / batch_size

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
            #  x_n = (x - μ) / sqrt(σ**2 + ε)
            x_normed = (x - mean) / (var + self.eps) ** 0.5

        # 测试时使用动量机制
        else:
            # x_n = (x - I_mean) / sqrt(I_std ** 2 + ε)
            x_normed = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5

        #  y = γ*x_n + β
        return x_normed * self.weight + self.bias

        
'''
    以概率p将输入值清零
    输出放大: x = x / (1 - p)
    推理时不进行操作
'''
class Dropout(Module):
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # 以p的概率生成0, randb参数p是生成1的概率,因此传入1 - self.p
            mask = init.randb(*x.shape, p = 1 - self.p , dtype='float32', device=x.device)
            x = x * mask
            z = x / (1 - self.p)
        # 训练时已经将x放大以与推理对齐
        else:
            z = x
        return z
    


class Residual(Module):
    
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x) 