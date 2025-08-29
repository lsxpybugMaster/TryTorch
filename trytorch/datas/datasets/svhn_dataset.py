import numpy as np
from scipy.io import loadmat
from typing import List, Optional, Iterable
from ..data_basic import Dataset


class SVHNDataset(Dataset):
    def __init__(
        self,
        file: str,
        transforms: Optional[List] = None,
    ):
        # 需要transforms,所以继承父类__init__
        super().__init__(transforms)

        data = loadmat(file)
        self.images = data['X']
        self.labels = data['y'].flatten()

    
    def __getitem__(self, index) -> object:
        # 取多个数据
        if(isinstance(index, (Iterable, slice))):
            img = np.stack([
                i.reshape((1, 32, 32)) for i in self.images[index]
            ])

        else:
            img = self.images[index].reshape((1, 1, 32, 32))

        img = self.apply_transforms(img)

        label = self.labels[index]

        return img, label
    

    def __len__(self) -> int:
        return len(self.images)
