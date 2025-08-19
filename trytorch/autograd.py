'''
自动求导部分
- 定义计算图结构
'''

import numpy
import cupy

# 类型约束,不仅约束容器类型,同时约束类型中类型  如List[int]
from typing import List, Optional, Tuple, Union

from .array_device import *

# array_api包含一系列跨API的计算函数
# NDArray是一个类型 : null | numpy.ndarray | cupy.ndarray
from .array_api import array_api, NDArray

# 运算操作 计算图边
# 虚基类
class Op:
    
    # 使用Op_instance_name()进行调用
    def __call__(self, *args):
        raise NotImplementedError()
    
    # 前向计算的入口
    def compute(self,*args: Tuple[NDArray]):
        '''
        计算前向传播
        '''
        raise NotImplementedError()
    
    # 可以返回Value或者Value元组
    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]: 
        '''
        计算梯度
        '''
        raise NotImplementedError()
        

    # 一定返回Value元组
    # 反向传播的入口
    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        # 不是元组需要打包成元组
        output = self.gradient(out_grad,node)
        if isinstance(output,tuple):
            return output
        elif isinstance(output,list):
            return tuple(output)
        else:
            return (output,)


# 计算图节点
class Value:
    
    '''
    ⚠️以下都是类型注解
    不起任何作用,当作注释
    '''
    # 算子(可为null)  
    # op会对所有inputs值做计算得到cached_data
    op: Optional[Op]

    # 输入节点 用字符串表对象 需要是Value节点列表
    inputs: List["Value"]

    # 存储的数据
    cached_data: NDArray
    
    requires_grad: bool

    # 计算该节点的值
    def realize_cached_data(self):
        # 如果该节点有值,说明之前已经计算过一次,无需再算
        if self.cached_data is not None:
            return self.cached_data
        
        # 否则需要进行计算,构建该节点到计算图中
        # '*'用于解包列表为参数送给函数
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs] 
        )
        return self.cached_data

    # 是否为叶节点(数据入口),无op
    def is_leaf(self):
        return self.op is None
    

    # 工厂方法
    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"], 
        *,  # 分隔符：之后的参数必须用关键字传递
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        # 以下是真正变量！
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad


# 张量,计算图节点,并且含有梯度!
# 真正的图节点
class Tensor(Value):

    # ⚠️类型注解,不起任何作用,当作注释
    grad: "Tensor"

    # 初始化节点值self.cached_data
    def _set_cached_data(array,*,device,dtype):
        # Tensor的复制,需要判断类型来源是否相同(numpy与cupy)
        if isinstance(array,Tensor):
            device = array.device if device is None else device
            dtype  = array.dtype  if dtype  is None else dtype
            # 🐍 Python的if/else不创建新的作用域 cached_data变量作用于整个函数
            if device == array.device and dtype == array.dtype:
                # 类型相同直接赋值即可
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(array.numpy(),dtype=dtype) if device == cpu() else Tensor._array_from_cupy(array.cupy(),dtype=dtype)            
        # 非Tensor,赋值即可
        else:
            if device is None:
                device = gpu() if isinstance(array,(cupy.generic, cupy.ndarray)) else cpu()
            cached_data = Tensor._array_from_numpy(array,dtype=dtype) if device == cpu() else Tensor._array_from_cupy(array,dtype=dtype)

        return cached_data # cached_data 作用域仅限函数内


    # 示例初始化: x = trytorch.Tensor([1,2,3], dtype="float32")
    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype = None,
        requires_grad = True,
        **kwargs
    ):
        # STEP1 : 从array中获取cached_data值
        cached_data = self._set_cached_data(array,device=device,dtype=dtype)
        
        # STEP2 : 调用父类工厂函数
        self._init(
            None, # Op
            [],   # Inputs[Tensor]
            cached_data=cached_data,
            requires_grad=requires_grad
        )
    
    # 类型转换为numpy_array
    @staticmethod
    def _array_from_numpy(array, dtype):
        return numpy.array(array, dtype=dtype)
    
    # 类型转换为cupy_array
    @staticmethod
    def _array_from_cupy(array, dtype):
        return cupy.array(array,dtype=dtype)
       