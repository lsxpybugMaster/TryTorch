'''
基础的data utils
'''
import numpy as np
from ..autograd import Tensor

from typing import Optional, List


class Dataset:
    '''
        Dataset抽象类
    '''

    def __init__(self, transforms: Optional[List] = None):
        '''
            Args:
                transforms: List[transform] | None
        '''
        self.transforms = transforms


    def __getitem__(self, index) -> object:
        '''
            index 可以为一个batch的全部索引
        '''
        raise NotImplementedError
    

    def __len__(self) -> int:
        raise NotImplementedError
    

    def apply_transforms(self, x):
        if self.transforms is not None:
            for tform in self.transforms:
                x = tform(x)
        return x
    


class DataLoader:
    '''
        数据加载器 = 数据集 + 迭代器
    '''

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        # batch idx
        self.idx = 0

        # 有序索引/shuffle索引        
        '''
            依据batch切分数组
            e.g  len = 10  batch = 4
            索引: [0,1,2,3,4,5,6,7,8,9] (或suffle的)
            切分位置: range(4, 10, 4) = [4, 8,]
            结果: [array(0,1,2,3), array(4,5,6,7), array(8,9)]
        '''
        # 这里如果不shuffle,那么只需切片一次
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), # 被切片的索引
                range(batch_size, len(dataset), batch_size) #生成切分位置
            )

          
    def __iter__(self):
        '''返回一个迭代器对象,即返回自身'''
        # 每个epoch都重新初始化一次
        self.idx = 0

        # 如果shuffle,那么每个epoch都需要重新shuffle
        if self.shuffle:
            tmp_range = np.arange(len(self.dataset))
            np.random.shuffle(tmp_range)
            self.ordering = np.array_split(
                tmp_range, # 被切片的索引
                range(self.batch_size, len(self.dataset),self.batch_size) #生成切分位置
            ) 

        return self
    

    def __next__(self):
        if self.idx < len(self.ordering):
            data = self.dataset[self.ordering[self.idx]]
            self.idx += 1
            return [Tensor(x) for x in data]
        else:
            raise StopIteration
        
    
    def __len__(self):
        '''
            返回batch个数
        '''
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
