'''
自动求导部分
- 定义计算图结构
'''

# 类型约束,不仅约束容器类型,同时约束类型中类型  如List[int]
from typing import List, Optional, Tuple, Union


# 运算操作 计算图边
# 虚基类
class Op:
    
    # 使用Op_instance_name()进行调用
    def __call__(self, *args):
        raise NotImplementedError()
    
    
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
    
    # 算子(可为null)
    op: Optional[Op]

    # 输入节点 用字符串表对象 需要是Value节点列表
    inputs: List["Value"]

    # 存储的数据
    cached_data: NDArray
    
    requires_grad: bool

    # 计算该节点的值
    def realize_cached_data(self):
        pass
    

    # 是否为计算图终点,无op
    def is_leaf(self):
        return self.op is None
    

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
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad



