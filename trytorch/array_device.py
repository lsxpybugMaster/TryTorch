'''
定义设备类型
'''
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
