'''
定义设备类型
'''

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

