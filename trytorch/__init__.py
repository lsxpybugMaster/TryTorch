'''
__init__文件在用户导入文件夹(包)时初始化执行
仅执行一次
可以在这里定义一些简化导入路径
'''

# 将子文件中所有模块全部暴露
from .autograd import *
from .array_device import *
from .optim import *

# 
from . import ops
from . import init


# 之后逻辑也会在导入包时自动执行
# print("Import Trytorch Success")