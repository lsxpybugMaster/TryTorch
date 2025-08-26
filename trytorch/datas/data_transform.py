import numpy as np

'''
    数据增强类,其实例可作为List[Transform]传入dataset初始化
'''
class Transform:
    def __call__(self, x):
        raise NotImplementedError
    

class RandomFlipHorizontal(Transform):
    
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
            水平翻转图像
            Args:
                img: NDArray (N, C, H, W)
            Returns:
                依据概率p进行翻转的 img
        """

        flip_img = np.random.rand() < self.p
        if flip_img:
            img = img[:,:,:,::-1]
        return img



class RandomCrop(Transform):
    
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        '''
            随机裁减图像
            Args:
                img: NDArray (N, C, H, W)
            Returns:
                依据padding 进行裁剪后的 img (N, C, H, W)
        '''
        shift_x, shift_y = np.random.randint(low=-self.padding,high=self.padding+1,size=2)
        N, C, H, W = img.shape
        img_pad = np.zeros((N, C, H + 2 * self.padding , W + 2 * self.padding))
        img_pad[:, :, self.padding : H + self.padding, self.padding : W + self.padding] = img
        # 所取区域大小一定为(H, W)
        img_crop = img_pad[
            :, :, 
            self.padding + shift_x : H + self.padding + shift_x,
            self.padding + shift_y : W + self.padding + shift_y
        ]
        return img_crop



class Normalize(Transform):
    def __init__(self, mean, std):
        '''
            mean, std 为三个通道的数据
        '''
        self.mean = np.array(mean).reshape(1, 3, 1, 1)
        self.std  = np.array(std).reshape(1, 3, 1, 1)

    def __call__(self, img):
        '''
            图像的归一化处理:
            Args:
                img: NDArray (N, C, H, W)
            Returns:
                归一化图像
        '''
        # 将图像转换为浮点数，并归一化到 [0, 1] 范围
        img = img.astype(np.float32) / 255.0
        img -= self.mean
        img /= self.std
        return img