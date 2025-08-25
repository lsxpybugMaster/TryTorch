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

    算子书写规则:
        前向计算,使用array_api操作NDArray
        包装前向计算(Tensor计算)为用户接口func()
        反向传播,使用func()计算梯度
'''

# Tensor相加
class EWiseAdd(TensorOp):
    # 实现Value类的compute
    def compute(self, a: NDArray, b: NDArray):
        return a + b
    
    '''
    y = a + b   
    ∂y/∂a = 1  v¯{a-y} = grad_y * 1 = grad_y
    ∂y/∂b = 1  v¯{b-y} = grad_y * 1 = grad_y
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad
    

'''
调用链
EWiseAdd().__call__().make_from_op().realize_cached_data().compute()
'''
# 直观的用户接口
def add(a, b):
    return EWiseAdd()(a, b)
    


# Tensor 加 标量    与上面相比,梯度反传不同!
class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar
    
    '''
    y = a + C
    ∂y/∂a = 1    v¯{a-y} = grad_y * 1 = grad_y
    C是非张量,不需传播梯度
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)
    

def add_scalar(a, scalar):
    return AddScalar(scalar)(a)



# 减法算子直接使用add(a, -b)
# 只需提供负号算子
class Negate(TensorOp):
    def compute(self, a: NDArray):
        return -a
    '''
    y = -a
    ∂y/∂a = -1  v¯{a-y} = -grad_y 
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        return negate(out_grad)


def negate(a):
    return Negate()(a)



# Tensor 相乘
class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b
    
    '''
    y = a * b
    ∂y/∂a = b  v¯{a-y} = grad_y * b 
    ∂y/∂b = a  v¯{b-y} = grad_y * a 
    所以还需要知道a, b的节点值
    存储在y的input: List[Tensor] 结构中
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs    
        return b * out_grad, a * out_grad
    

def multiply(a, b):
    return EWiseMul()(a, b)



# Tensor 与 标量相乘    同理由于梯度反传不同所以需额外构建算子
class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    '''
    y = a * C
    ∂y/∂a = C   v¯{a-y} = grad_y * C
    C标量不反传梯度
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)  


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)



# Tensor 的 scalar 次幂
class PowerScalar(TensorOp):
    def __init__(self, scalar: int):
        self.scalar = scalar

    # array_api依据a类型选择numpy OR cupy
    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)
    
    '''
    y = a ** C
    ∂y/∂a = Ca ** (C - 1)  v¯{a-y} = grad_y * Ca ** (C - 1) 
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
    

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)



# Tensor间除法
class EWiseDiv(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a / b
    
    '''
    y = a / b
    ∂y/∂a = 1 / b           v¯{a-y} = grad_y * 1 / b 
    ∂y/∂b = - a / b**2      v¯{b-y} = grad_y * (- a / b**2 ) 
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return out_grad * (power_scalar(b, -1)), out_grad * (divide(negate(a), power_scalar(b, 2)))


def divide(a, b):
    return EWiseDiv()(a, b)



# Tensor 除以 标量
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar
    
    '''
    y = a / C
    ∂y/∂a = 1 / C           v¯{a-y} = grad_y * 1 / C 
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)



class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    # 通过交换矩阵shape参数"实现"转秩
    def compute(self, a: NDArray):
        # 未转置时维度 [1,2,3,4]
        self.axis_l = list(range(a.ndim))
        # 不提供参数默认转后两维
        if self.axes is None:
            self.axis_l[-1], self.axis_l[-2] = self.axis_l[-2], self.axis_l[-1]
        else:
            self.axis_l[self.axes[0]], self.axis_l[self.axes[1]] = self.axes[1], self.axes[0]
        
        return array_api.transpose(a, self.axis_l)
    
    '''
    许多线性操作的梯度传播就是该操作的逆操作。
    对上游梯度进行相同转置操作即实现传播
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)



class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.reshape(a, self.shape)

    '''
    许多线性操作的梯度传播就是该操作的逆操作。
    对上游梯度进行相同Reshape操作即实现传播
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return reshape(out_grad, a.shape)


def reshape(a, shape):
    return Reshape(shape)(a)



class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.broadcast_to(a, self.shape)

    '''
    许多线性操作的梯度传播就是该操作的逆操作。
    对上游梯度在广播增加的维度上进行求和，然后重塑回原始输入形状。
    e.g. [1, 2] -> [1, 2]   [g1, g2] -> [g1 + g3, g2 + g4]
                   [1, 2]   [g3, g4]
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        in_shape = node.inputs[0].shape
        out_shape = out_grad.shape

        # 对齐维度,先获取元组长度差别,再根据差别向前补1
        diff_len = len(out_shape) - len(in_shape)
        # 注意元组间计算直接是加形状(拼接)而非数值
        in_shape_aligned = (1,) * diff_len + in_shape

        # 寻找拼接的维度,对该维度求和
        axes = []
        for i, (in_dim, out_dim) in enumerate(zip(in_shape_aligned, out_shape)):
            if in_dim == 1 and out_dim > 1:
                axes.append(i)

        grad = out_grad.sum(axes=tuple(axes))
        # sum 可能会舍去维度, 我们手动恢复
        return grad.reshape(in_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)



class Summation(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None):
        self.axes = axes
        # 确保一定为tuple
        if isinstance(self.axes, int):
            self.axes = tuple([self.axes])

    def compute(self, a: NDArray):
        # numpy 默认 keepdim = False
        return array_api.sum(a, self.axes)
    
    '''
    许多线性操作的梯度传播就是该操作的逆操作。
    在反向传播时,上游梯度会被均匀分配到所有被求和的元素。
    e.g. [1, 2, 3] => [6,    [g1, => [g1  g1  g1]
         [4, 5, 6]     15]    g2]    [g2  g2  g2]
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        final_shape = list(input_shape)

        # 还原被压缩的维度
        # e.g.    sum[(2,3) axes = 1] -> (2) => (2, 1)  
        if self.axes:
            for dim in self.axes:
                final_shape[dim] = 1
        # numpy中axes = None 则求全部和并返回标量
        # e.g.   sum[(2,3) axes = None] -> _ => (1, 1)  
        else:
            final_shape = [1 for _ in range(len(final_shape))]

        # 完成形状转换
        out_grad = reshape(out_grad, final_shape)

        # 广播grad
        # e.g. [g1, g2] * [1  1  1] = [g1 g1 g1]
        #                 [1  1  1]   [g2 g2 g2]
        return out_grad * array_api.ones(input_shape,dtype="float32",device=out_grad.device)
    

def summation(a, axes=None):
    return Summation(axes)(a)



# 矩阵乘法
class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return array_api.matmul(a, b)
    
    '''
    y = A @ B   
    A: (m, k)  
    B: (k, n) 
    y, grad  (m, n)
    ∂y/∂A = B^T          : (n, k)          
    ∂y/∂B = A^T          : (k, m)      
    gradA = grad @ B^T   : (m, k)
    gradB = A^T  @ grad  : (k, n)
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        A, B = node.inputs
        gradA = matmul(out_grad, transpose(B))
        gradB = matmul(transpose(A), out_grad)

        # 处理广播机制带来的维度不匹配问题。
        if gradA.shape != A.shape:
            gradA = summation(gradA, tuple(range(len(gradA.shape) - len(A.shape))))
        if gradB.shape != B.shape:
            gradB = summation(gradB, tuple(range(len(gradB.shape) - len(B.shape))))

        return gradA, gradB


def matmul(a, b):
    return MatMul()(a, b)



# log函数
class Log(TensorOp):
    def compute(self, a: NDArray):
        return array_api.log(a)
    
    '''
    y = ln(a)  
    ∂y/∂a = 1 / a   
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return out_grad * power_scalar(a, -1)


def log(a):
    return Log()(a)



# sin函数
class Sin(TensorOp):
    def compute(self, a: NDArray):
        return array_api.sin(a)
    '''
    y = sin(a)  
    ∂y/∂a = cos(a)   
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        '''
            ❌ 严重错误: return out_grad * array_api.cos(a)
            array_api是操作NDArray的,用于实现前向计算中数值的真正计算
            进行Tensor运算,必须使用ops中封装好的函数
        '''
        return out_grad * cos(a)


def sin(a):
    return Sin()(a)



class Cos(TensorOp):
    def compute(self, a: NDArray):
        return array_api.cos(a)

    '''
    y = cos(a)  
    ∂y/∂a = -sin(a) = sin(-a)
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return out_grad * sin(negate(a))


def cos(a):
    return Cos()(a)



# 自然指数函数
class Exp(TensorOp):
    def compute(self, a: NDArray):
        return array_api.exp(a)
    
    '''
    y = exp(a)
    ∂y/∂a = exp(a)    v¯{a-y} = grad_y * exp(a)  
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return (out_grad * exp(a),)
    

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

    def compute(self, Z: NDArray):
        # 用于广播,因此保留维度
        max_Z = array_api.max(Z,axis=self.axes, keepdims=True)
        # 用于与结果相加,因此舍去维度
        max_z = array_api.max(Z,axis=self.axes, keepdims=False)
        
        logsumexp = max_z + array_api.log(
            array_api.sum(array_api.exp(Z - max_Z),axis=self.axes)
        )

        return logsumexp
    
    '''
    TODO: 暂时搁置这部分计算,后续弄懂
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        max_Z = array_api.max(Z.cached_data, axis=self.axes, keepdims=True)
        exp_val = exp(Z - Tensor(max_Z))
        sum_val = summation(exp_val, axes=self.axes)

        log_grad = out_grad / sum_val
        
        input_shape = node.inputs[0].shape
        final_shape = list(input_shape)
        if self.axes:
            if isinstance(self.axes, int):
                final_shape[self.axes] = 1
            else:
                for dim in self.axes:
                    final_shape[dim] = 1
        else:
            final_shape = [1 for _ in range(len(final_shape))]
        sum_grad = reshape(log_grad, tuple(final_shape))
        sum_grad_b = broadcast_to(sum_grad, Z.shape)
        return exp_val * sum_grad_b


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)



class ReLU(TensorOp):
    def compute(self, a: NDArray):
        return array_api.maximum(a, 0)

    '''
    y = max(a, 0)
    ∂y/∂a = 1  (a > 0) 
          = 0  (a <= 0)
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        dReLU = array_api.where(a.cached_data > 0, 1.0, 0.0).astype("float32")
        return out_grad * dReLU


def relu(a):
    return ReLU()(a)



class Sigmoid(TensorOp):
    def compute(self, a: NDArray):
        return 1 / (1 + array_api.exp(-a))
    
    '''
    y = sig(a)
    ∂y/∂a = sig(a)(1 - sig(a))       
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return out_grad * sigmoid(a) * (1 - sigmoid(a))
    

def sigmoid(a):
    return Sigmoid()(a)



class Tanh(TensorOp):
    def compute(self, a: NDArray):
        return array_api.tanh(a)
    
    '''
    y = tanh(a)
    ∂y/∂a = 1 - tanh(a)**2
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return out_grad * (1 + (-tanh(a) ** 2))


def tanh(a):
    return Tanh()(a)



class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        return array_api.flip(a, self.axes)
    
    '''
    许多线性操作的梯度传播就是该操作的逆操作。
    直接将梯度翻转即可
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        return flip(out_grad, self.axes)
    

def flip(a, axes):
    return Flip(axes)(a)



def compact(array):
    out_array = array.copy()
    return out_array
