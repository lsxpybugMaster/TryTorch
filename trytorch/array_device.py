'''
定义设备类型
'''
import numpy
import cupy

# 虚基类
class Device():
    pass

class CPUDevice(Device):
    # 与__str__类似
    def __repr__(self):
        return "trytorch.cpu()"
    
    def __hash__(self):
        return self.__repr__().__hash__()
    
    def __eq__(self, other):
        return isinstance(other, CPUDevice)
    
    def enabled(self):
        return True
    
def cpu():
    return CPUDevice()


class GPUDevice(Device):
    def __repr__(self):
        return "trytorch.gpu()"
    
    def __hash__(self):
        return self.__repr__().__hash__()
    
    def __eq__(self, other):
        return isinstance(other, GPUDevice)
    
    def enabled(self):
        if cupy.cuda.runtime.getDeviceCount() > 0:
            print("GPU is available")
        else:
            print("GPU is not available")


def gpu():
    return GPUDevice()



def get_device_by_data(data):
    '''
        根据数据获取当前的device
    '''
    if isinstance(data, (numpy.generic, numpy.ndarray)):
        return cpu()
    elif isinstance(data, (cupy.generic, cupy.ndarray)):
        return gpu()
    else:
        raise ValueError(f"Unknown array types: {type(data).__module__}")