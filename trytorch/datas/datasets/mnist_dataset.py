from typing import List, Optional, Iterable
from ..data_basic import Dataset
import numpy as np
import gzip


class MNISTDataset(Dataset):
    
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(image_filename, 'rb') as f:
            image_data = np.frombuffer(f.read(), np.uint8, offset=16)
        
        with gzip.open(label_filename, 'rb') as f:
            label_data = np.frombuffer(f.read(), np.uint8, offset=8)

        X = image_data.reshape(-1, 784).astype(np.float32) / 255
        
        self.images = X
        self.labels = label_data
        self.transform = transforms

    ### 所有dataset子类必须实现__getitem__与__len__方法

    def __getitem__(self, index) -> object:
        '''同时处理多个索引与批量索引的情况'''
        if isinstance(index, (Iterable, slice)):
            img = np.stack([
                i.reshape((1, 28, 28)) for i in self.images[index]
            ])
        else:
            img = self.images[index].reshape((1, 28, 28))
        
        if self.transform:
            for trans in self.transform:
                img = trans(img)
        return img, self.labels[index]
    

    def __len__(self) -> int:
        '''返回样本数量'''
        return self.images.shape[0]