'''
选择底层计算库
- cpu : numpy
- gpu : cupy
- pytorch中是动态派发至C++计算
'''

import numpy
import cupy
# 从当前路径查找,防止查找到根目录下同名模块
from .array_device import Device, cpu, gpu

from typing import Optional, Union

# 一个类型别名,该类型可以是 null | numpy.ndarray | cupy.ndarray
NDArray = Optional[Union[numpy.ndarray, cupy.ndarray]]

class array_api:
    '''
    根据设备或计算数据的形式选择对应的底层计算API
    由于numpy与cupy API相同,才可以如此实现
    '''

    @staticmethod
    def _backend_from_array(data):
        if type(data).__module__ == numpy.__name__:
            return numpy
        elif type(data).__module__ == cupy.__name__:
            return cupy
        else:
            raise ValueError(f"Unknown array types: {type(data).__module__}")
        
        
    @staticmethod
    def _backend_from_device(device: Device):
        if device == cpu(): # Device实现了__eq__
            return numpy
        elif device == gpu():
            return cupy
        else:
            raise ValueError(f"Unknown device: {type(device)}")
        

    '''
    包装底层API,依据[计算数据类型]自动选择底层API
    包含常用函数,如reshape,sum等
    '''
    @staticmethod 
    def add(a, b):
        return array_api._backend_from_array(a).add(a,b)
    
    @staticmethod
    def where(condition, x, y):
        return array_api._backend_from_array(condition).where(condition, x, y)

    @staticmethod
    def multiply(a, b):
        return array_api._backend_from_array(a).multiply(a, b)
    
    @staticmethod
    def divide(a, b):
        return array_api._backend_from_array(a).divide(a, b)
    
    @staticmethod
    def power(a, exponent):
        return array_api._backend_from_array(a).power(a, exponent)
    
    @staticmethod
    def transpose(a, axes=None):
        return array_api._backend_from_array(a).transpose(a, axes)
    
    @staticmethod
    def reshape(a, newshape):
        return array_api._backend_from_array(a).reshape(a, newshape)
    
    @staticmethod
    def sum(a, axis=None, keepdims=False):
        return array_api._backend_from_array(a).sum(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def broadcast_to(a, shape):
        return array_api._backend_from_array(a).broadcast_to(a, shape)
    
    @staticmethod
    def matmul(a, b):
        return array_api._backend_from_array(a).matmul(a, b)
    
    @staticmethod
    def log(a):
        return array_api._backend_from_array(a).log(a)
    
    @staticmethod
    def exp(a):
        return array_api._backend_from_array(a).exp(a)
    
    @staticmethod
    def sin(a):
        return array_api._backend_from_array(a).sin(a)
    
    @staticmethod
    def cos(a):
        return array_api._backend_from_array(a).cos(a)
    
    @staticmethod
    def negative(a):
        return array_api._backend_from_array(a).negative(a)
    
    @staticmethod
    def maximum(a, b):
        return array_api._backend_from_array(a).maximum(a, b)
    
    @staticmethod
    def max(a, axis=None, keepdims=False):
        return array_api._backend_from_array(a).max(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def minimum(a, b):
        return array_api._backend_from_array(a).minimum(a, b)
    
    @staticmethod
    def min(a, axis=None, keepdims=False):
        return array_api._backend_from_array(a).min(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def abs(a):
        return array_api._backend_from_array(a).abs(a)
    
    @staticmethod
    def sqrt(a):
        return array_api._backend_from_array(a).sqrt(a)
    
    @staticmethod
    def random_normal(shape):
        return array_api._backend_from_array(shape).random.normal(shape)
    
    @staticmethod
    def random_uniform(shape):
        return array_api._backend_from_array(shape).random.uniform(shape)
    
    @staticmethod
    def as_strided(a, shape, strides):
        return array_api._backend_from_array(a).lib.stride_tricks.as_strided(a, shape, strides)
    
    @staticmethod
    def pad(a, pad_width, mode='constant', **kwargs):
        return array_api._backend_from_array(a).pad(a, pad_width, mode=mode, **kwargs)
    
    @staticmethod
    def flip(a, axis):
        return array_api._backend_from_array(a).flip(a, axis)
    
    @staticmethod
    def tanh(a):
        return array_api._backend_from_array(a).tanh(a)
    
    '''
    包装底层API,依据[设备类型]自动选择底层API
    包含常用函数,如ones,randn等
    '''
    @staticmethod
    def ones(shape, dtype = "float32", device = cpu()):
        return array_api._backend_from_device(device).ones(shape,dtype= dtype)

    @staticmethod
    def zeros(shape, dtype = "float32", device = cpu()):
        return array_api._backend_from_device(device).zeros(shape, dtype=dtype)
    
    @staticmethod 
    def arange(start, stop=None, step=1, dtype="float32", device=cpu()):
        api = array_api._backend_from_device(device)
        if stop is None:
            return api.arange(start, dtype=dtype)
        else: 
            return api.arange(start, stop, step, dtype)
 
    @staticmethod
    def rand(*shape, device = cpu()):
        return array_api._backend_from_device(device).random.rand(*shape)
        
    @staticmethod
    def randn(*shape, device = cpu()):
        return array_api._backend_from_device(device).random.randn(*shape)

    @staticmethod
    def one_hot(n, i, dtype = "float32", device = cpu()):
        return array_api._backend_from_device(device).eye(n, dtype=dtype)[i]
    