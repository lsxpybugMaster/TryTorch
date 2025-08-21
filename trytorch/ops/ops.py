from typing import Optional, Union
from ..autograd import Tensor, TensorOp
from ..array_api import array_api, NDArray
from ..array_device import cpu, gpu
import numpy
import cupy


'''
    所有算子继承TensorOp,实现前向计算和梯度计算(反向传播)
    前向过程调用链:
    生成TensorOp子类的实例 -> 
    调用__call__函数 -> 
    调用make_from_op -> 
    调用realize_cached_data -> 
    调用op.compute完成计算
'''

# Tensor相加
class EWiseAdd(TensorOp):
    # 实现Value类的compute
    def compute(self, a: NDArray, b: NDArray):
        return a + b
    
'''
调用链
EWiseAdd().__call__().make_from_op().realize_cached_data().compute()
'''
# 直观的用户接口
def add(a, b):
    return EWiseAdd()(a, b)
    


# Tensor 加 标量    与上面相比,梯度反传不同!
class AddScalar(TensorOp):
    def __init__(self,scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar
    
def add_scalar(a, scalar):
    return AddScalar(scalar)(a)



# 减法算子直接使用add(a, -b)
# 只需提供负号算子
class Negate(TensorOp):
    def compute(self, a):
        return -a

def negate(a):
    return Negate()(a)



# Tensor 相乘
class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b
    
def multiply(a, b):
    return EWiseMul()(a, b)



# Tensor 与 标量相乘    同理由于梯度反传不同所以需额外构建算子
class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar
    

def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)



# Tensor 的 scalar 次幂
class PowerScalar(TensorOp):
    def __init__(self, scalar: int):
        self.scalar = scalar

    # array_api依据a类型选择numpy OR cupy
    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)
        
    
def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)



# Tensor间除法
class EWiseDiv(TensorOp):
    def compute(self, a, b):
        return a / b
    

def devide(a, b):
    return EWiseDiv()(a, b)



# Tensor 除以 标量
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar
    

def devide_scalar(a, scalar):
    return DivScalar(scalar)(a)



class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    # 通过交换矩阵shape参数"实现"转秩
    def compute(self, a):
        # 未转置时维度 [1,2,3,4]
        self.axis_l = list(range(a.ndim))
        # 不提供参数默认转后两维
        if self.axes is None:
            self.axes_l[-1], self.axis_l[-2] = self.axis_l[-2], self.axis_l[-1]
        else:
            self.axis_l[self.axes[0]], self.axis_l[self.axes[1]] = self.axes[1], self.axes[0]
        
        return array_api.transpose(a, self.axis_l)


def transpose(a, axes=None):
    return Transpose(axes)(a)



class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)


def reshape(a, shape):
    return Reshape(shape)(a)



class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)



class Summation(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None):
        self.axes = axes
        # 确保一定为tuple
        if isinstance(self.axes, int):
            self.axes = tuple([self.axes])

    def compute(self, a):
        return array_api.sum(a, self.axes)
    

def summation(a, axes=None):
    return Summation(axes)(a)



# 矩阵乘法
class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)
    

def matmul(a, b):
    return MatMul()(a, b)


# log函数
class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)
    

def log(a):
    return Log()(a)



# 自然指数函数
class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)
    

def exp(a):
    return Exp()(a)

'''
f(x) = log(e_x1 + e_x2 + ... + e_xn)

为了数值稳定,防止exp(xi)溢出,实际采用数值稳定的形式
f(x) = max_z + log{e_(x1 - max_Z) + ... + e_(xn - max_Z)}
'''
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        # 用于广播,因此保留维度
        max_Z = array_api.max(Z,axis=self.axes, keepdims=True)
        # 用于与结果相加,因此舍去维度
        max_z = array_api.max(Z,axis=self.axes, keepdims=False)
        
        logsumexp = max_z + array_api.log(
            array_api.sum(array_api.exp(Z - max_Z),axis=self.axes)
        )

        return logsumexp


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)



class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)


def relu(a):
    return ReLU()(a)



class Sigmoid(TensorOp):
    def compute(self, a):
        return 1 / (1 + array_api.exp(-a))
    

def sigmoid(a):
    return Sigmoid()(a)



class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)


def tanh(a):
    return Tanh()(a)



class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)
    

def flip(a, axes):
    return Flip(axes)(a)



def compact(array):
    out_array = array.copy()
    return out_array
