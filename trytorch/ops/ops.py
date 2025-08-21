from typing import Optional
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

# 算子
class EWiseAdd(TensorOp):
    # 实现Value类的compute
    def compute(self, a: NDArray, b: NDArray):
        return a + b
    
'''
调用链
EWiseAdd().__call__().make_from_op().realize_cached_data().compute()
'''
def add(a, b):
    return EWiseAdd()(a, b)
    