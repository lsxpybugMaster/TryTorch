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
