from typing import Optional, Union
from ..autograd import Tensor, TensorOp
from ..array_api import array_api, NDArray
from ..array_device import get_device_by_data
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
def add(a: Tensor, b: Tensor):
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



#---------------------------卷积相关算子--------------------------

class Dilate(TensorOp):
    '''
        在矩阵元素之间补充零,该操作在利用反卷积进行反向传播时出现
        Args:
            axes: 需要补充0的维度 
            dilation: 元素之间补充0的个数
    '''
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation


    def compute(self, a: NDArray):
        shape = a.shape
        out_shape = list(shape)
        
        # 🐍 在Python里,slice是一个内置类,表示切片操作。 2:10:2 等价于 slice(2,10,2)
        # 初始先默认切片切完所有数据 slices = [slice(0, shape[0], None), ...]
        slices = [slice(0, out_shape[idx]) for idx in range(len(shape))]

        for ax in self.axes:
            if ax >= len(out_shape):
                continue
            # 膨胀对应维度
            out_shape[ax] = out_shape[ax] * (1 + self.dilation)
            # 那么该维度的切片需要增加步长: slice(0, shape[ax]) => slice(0, shape[ax], 1 + dilation) 实现跳步切片
            slices[ax] = slice(0, out_shape[ax], 1 + self.dilation)
        
        # 调用array_device模块下函数确定device
        device = get_device_by_data(a)

        # 预先构建膨胀后的全0矩阵
        out_tensor = array_api.zeros(out_shape, dtype="float32", device=device)

        # 使用切片索引复制数组a到正确位置
        out_tensor[tuple(slices)] = a
        return out_tensor
    
    '''
    许多线性操作的梯度传播就是该操作的逆操作。
    对上游梯度进行相同转置操作即实现传播
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        out_grad = undilate(out_grad, self.axes, self.dilation)
        return out_grad


def dilate(a, axes: tuple, dilation: int):
    return Dilate(axes, dilation)(a)
        


class UnDilate(TensorOp):
    
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):

        shape = a.shape
        
        # 默认先进行完整切片
        slices = [slice(0, shape[idx]) for idx in range(len(shape))]
    
        for ax in self.axes:
            if ax >= len(shape):
                continue
            
            # 原来写成了slice导致错误,注意变量的拼写!
            slices[ax] = slice(0, shape[ax], 1 + self.dilation)
        
        # 反向操作时切片就是对应的值
        # compact确保切片后数据在内存中连续
        return compact(a[tuple(slices)])
    
    '''
    许多线性操作的梯度传播就是该操作的逆操作。
    对上游梯度进行相同转置操作即实现传播
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        out_grad = dilate(out_grad, self.axes, self.dilation)
        return out_grad


def undilate(a, axes: tuple, dilation: int):
    return UnDilate(axes, dilation)(a)


# 卷积算子
class Conv(TensorOp):
    
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

   
    def compute(self, X: NDArray, W: NDArray):
        '''
            X: 特征图 (N, H , W, C)
            W: 卷积核 (K, _ , C_in, C_out)
            使用 img2col 将 X, W 重组
            Y = X @ W 
        '''

        N, _H, _W, C_in  = X.shape
        # (卷积核大小, _ , 输入通道数, 输出通道数)
        K, _,  I, C_out = W.shape  

        assert C_in == I, "卷积核的输入通道与特征图的通道数不匹配!"

        # 仅在H, W维度做padding
        pad_width = [
            (0,0),    
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0,0),
        ]
        # 得到padding后的特征图X
        X_pad = array_api.pad(X, pad_width, mode='constant', constant_values = 0) if self.padding > 0 else X

        '''
            img2col
            涉及多通道,多卷积核时:
            特征图通道横向拼接
            卷积核个数纵向拼接
        '''
        # 计算卷积核展开后的第一维度,即卷积核大小(K * K)与输入通道数 
        inner_dim = K * K * C_in
        
        # 获取padding的步幅, 步幅是一个元组，表示在内存中沿着每个维度前进一个元素时需要跳过的字节数。它描述了张量在内存中的布局方式
        Ns, Hs, Ws, Cs = X_pad.strides
        
        # 使用公式计算输出特征图的维度
        H_out = (_H - K + 2 * self.padding) // self.stride + 1
        W_out = (_W - K + 2 * self.padding) // self.stride + 1
        
        '''
            使用as_strided展开特征图
            as_strided不复制数据,而是通过改变对现有内存的解释方式来创建新的张量视图。
            shape 指定对该数据的解释方式
            strides 指定每次切换下一个数据时的步幅(与卷积步幅区分!)
        '''
        _X = array_api.as_strided(
            X_pad,
            shape = (N, H_out, W_out, K, K, C_in), # 创建所有可能的卷积窗口
            strides=(Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs) # 卷积核移动有步长,所以其中两维是需要额外计算步长的
        ) 

    
        _X_ = compact(_X).reshape((-1, inner_dim))
        
        _W_ = compact(W).reshape((-1, C_out))

        out = _X_ @ _W_

        return out.reshape((N, H_out, W_out, C_out))


    def gradient(self, out_grad: Tensor, node: Tensor):
        '''
            卷积反向传播:反卷积公式      
        '''
        X, W = node.inputs


        '''
            对 X 的导数 (用于链接梯度的反向传播)
            δx = pad(dilate(δz)) * rotate180(W)
        '''
        # padding部分已内置在conv中,所以先dilate
        
        # grad: (N, H, W, C)
        grad_dilate = dilate(out_grad, (1,2), self.stride - 1)

        # W: (K, K, Cin, Cout)
        W_r180 = flip(W, (0,1))

        W_r180_T = transpose(W_r180)

        K = W_r180.shape(0)
        
        # 反卷积
        grad_X = conv(grad_dilate, W_r180_T, 1, K - 1 - self.padding)


        '''
            对 W 的导数 (用于更新参数)
            δw = X * dilate(δz)
        '''
        # grad: (N, H, W, C) -> (W, H, N, C) -> (H, W, N, C)
        grad_dilate = grad_dilate.transpose((0, 2)).transpose((0, 1))
        
        # X : (N, H, W, C) -> (W, H, N, C)
        X_t = transpose(X, (0, 3))

        # 反卷积
        grad_W = conv(X_t, grad_dilate, 1, self.padding)

        grad_W = grad_W.transpose((0, 2)).transpose((0, 1))

        return Tensor(grad_X), grad_W



def conv(a, b, stride=1, padding=1):
    return Conv(stride,padding)(a, b)



