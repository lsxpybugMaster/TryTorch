import trytorch as torch
from trytorch.array_api import array_api

'''
一些初始化向量的函数
'''


def constant(*shape, c = 1.0,device=None, dtype="float32", requires_grad=False):
    '''生成shape形状的全c的张量'''
    device = torch.cpu() if device is None else device
    array  = array_api.ones(shape, device=device, dtype=dtype) * c
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def ones(*shape, device= None, dtype="float32", requires_grad=False):
    return constant(
        *shape, c = 1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )

def zeros(*shape, device= None, dtype="float32", requires_grad=False):
    return constant(
        *shape, c = 0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )