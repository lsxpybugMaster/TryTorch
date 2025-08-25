import trytorch as torch
from trytorch.array_api import array_api
import math
'''
一些初始化向量的函数
'''

# FIXME 此处用装饰器修改原始代码,里面可能有问题

# 装饰器
def tensor_init(func):
    '''
        考虑到包装这些array_api中的函数的方式相同,因此使用装饰器优化
        包含确定device,以及将得到的array_api形式的数据封装为Tensor的功能
    '''
    def wrapper(*shape, device=None, dtype="float32", requires_grad=False, **kwargs):
        device = torch.cpu() if device is None else device
        # 真正的函数调用: 返回得到的NDArray数组
        array = func(*shape, device=device, **kwargs)

        return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
    return wrapper


@tensor_init
def rand(*shape, low=0.0, high=1.0, **kwargs):
    return array_api.rand(*shape, **kwargs) * (high - low) + low


@tensor_init
def randn(*shape, mean=0.0, std=1.0, **kwargs):
    return array_api.randn(*shape, **kwargs) * std + mean


@tensor_init
def constant(*shape, c=1.0, **kwargs):
    '''生成shape形状的全C的张量'''
    return array_api.ones(shape, **kwargs) * c


@tensor_init
def ones(*shape, **kwargs):
    return array_api.ones(shape, **kwargs) 


@tensor_init
def zeros(*shape, **kwargs):
    return array_api.zeros(shape, **kwargs)   


# 装饰器仅支持装饰返回float32类型的Tensor,randb返回bool型,因此无法使用装饰器
def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    device = torch.cpu() if device is None else device
    array = array_api.rand(*shape, device=device) <= p
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


@tensor_init
def one_hot(n, i, **kwargs):
    return array_api.one_hot(n, i.realize_cached_data(), **kwargs)


def xavier_uniform(fan_in, 
                   fan_out, 
                   shape=None, 
                   device=None,
                   **kwargs):
    '''
        fan_in, fan_out : 输入,输出维度
        bound = sqrt(6 / (fan_in + fan_out))
        W ~ U(-bound, bound)    
    '''
    bound = math.sqrt(6 / (fan_in + fan_out))
    if shape is not None:
        return rand(*shape, low=-bound, high=bound, device=device, **kwargs)
    else:
        return rand(fan_in, fan_out, low=-bound, high=bound, device=device, **kwargs)


def kaiming_uniform(fan_in, 
                    fan_out, 
                    shape=None, 
                    nonlinearity="relu",
                    device=None,
                    **kwargs):
    '''
        fan_in : 输入维度
        bound = sqrt(6 / fan_in)
        W ~ U(-bound, bound)    
    '''
    assert nonlinearity == 'relu', 'Only relu supported currently'
    gain = math.sqrt(2) # 不同激活函数需不同gain值
    bound = gain * math.sqrt(3 / fan_in)
    if shape is not None:
        return rand(*shape, low=-bound, high=bound, device=device, **kwargs)
    else:
        return rand(fan_in, fan_out, low=-bound, high=bound, device=device, **kwargs)
    