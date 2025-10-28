import logging
import math
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .randaugment  import RandAugmentMC
from PIL import ImageFile

from augment.line import SobelEdgeDetector,CannyEdgeDetector,LaplacianEdgeDetector,LoGEdgeDetector,PrewittEdgeDetector,ScharrEdgeDetector,RobertsEdgeDetector,DOGEdgeDetector



Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

img_size = 224

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    
    labeled_idx = []
    unlabeled_idx = np.arange(len(labels))
    
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        if len(idx) < label_per_class:
            warning_msg = f"Warning: Not enough data for class {i}, only {len(idx)} available but {label_per_class} needed."
            print(warning_msg)
            # 可以选择使用所有可用数据，或者调整 label_per_class
            label_per_class_for_this_class = len(idx)
        else:
            label_per_class_for_this_class = label_per_class
        
        selected_idx = np.random.choice(idx, label_per_class_for_this_class, replace=False)
        labeled_idx.extend(selected_idx)
    
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, labeled_idx)
    
    # 检查是否选择了正确数量的标签数据
    assert len(labeled_idx) <= args.num_labeled, f"Error: Selected more labeled indices ({len(labeled_idx)}) than specified ({args.num_labeled})."
    
    # 如果需要，进行数据扩增（注意：这里的数据扩增逻辑可能需要根据实际情况调整）
    if args.expand_labels or len(labeled_idx) < args.batch_size * args.eval_step:
        num_expand_times = math.ceil(args.batch_size * args.eval_step / len(labeled_idx))
        expanded_labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_times)])
        
        # 如果扩增后的数据仍然不足以满足一个batch，则发出警告（或采取其他措施）
        if len(expanded_labeled_idx) < args.batch_size:
            warning_msg = f"Warning: Expanded labeled indices ({len(expanded_labeled_idx)}) are still less than one batch size ({args.batch_size})."
            print(warning_msg)
        
        # 使用扩增后的数据（或根据实际情况选择是否使用）
        labeled_idx = expanded_labeled_idx[:args.num_labeled]  # 如果扩增后超过了指定的标签数量，则截取前args.num_labeled个
        np.random.shuffle(labeled_idx)  # 打乱扩增后的标签数据（如果扩增了的话）
    else:
        np.random.shuffle(labeled_idx)  # 打乱原始标签数据
    
    # 确保最终选择的标签数量不超过指定的数量（虽然前面的逻辑已经保证了这一点，但这里再次检查以增加健壮性）
    labeled_idx = labeled_idx[:args.num_labeled]
    
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize((img_size, img_size)),

            transforms.RandomHorizontalFlip(),               # 随机水平翻转增强
            #transforms.RandomCrop(size=img_size,
                                  #padding=int(img_size * 0.125),
                                  #padding_mode='reflect'),   # 随机裁剪增强
        
           
           ])   
        self.strong = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),# 多裁剪随机增强（n=2个增强操作，m=10强度等级）
            #SobelEdgeDetector(dx=1, dy=0), # 边缘检测Sobel算子
            #CannyEdgeDetector(threshold1=50, threshold2=150),#边缘检测Canny算子
            #LaplacianEdgeDetector(ddepth=cv2.CV_64F, ksize=3),#边缘检测Laplacian算子
            #LoGEdgeDetector(kernel_size=5),  # 使用LoG边缘检测
            #PrewittEdgeDetector(),
            #ScharrEdgeDetector(threshold=128)  # 应用 Scharr 边缘检测
            RobertsEdgeDetector(threshold=128)  # 应用 Roberts 边缘检测
            #DOGEdgeDetector(kernel_size1=5, sigma1=1.0, kernel_size2=7, sigma2=2.0),
            
            
       
        ])
                   
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

# 构建分类模型的数据读取Dataset子类–MyDataset类
class train_chinesepaintings(Dataset):
    ''' 初始化文件路径或文件名列表。也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        imgs_path:a txt file, each line in the form of "image_path label" '''

    def __init__(self, imgs_path, idx=None, transform=None):
        self.imgs_path = imgs_path
        self.idx = idx
        self.transform = transform
        self.size = 0
        self.imgs_list = []

        file = open(self.imgs_path)  # 我们使⽤open()函数来打开⼀个⽂件, 获取到⽂件句柄.
        for f in file:
            self.imgs_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.imgs_list[idx].split(' ')[0]  # split(' ')按照空格分隔imgs_list [n]:表示选取第n个分片
        image = Image.open(image_path).convert('RGB') # 读取图片
        if(image.mode == 'L'):
             image = image.convert('RGB')
        label = int(self.imgs_list[idx].split(' ')[1])
        #image = np.array(image)
        label = np.array(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# 构建分类模型的数据读取Dataset子类–MyDataset类
class val_chinesepaintings(Dataset):
    ''' 初始化文件路径或文件名列表。也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        imgs_path:a txt file, each line in the form of "image_path label" '''

    def __init__(self, imgs_path, idx=None, transform=None):
        self.imgs_path = imgs_path
        self.idx = idx
        self.transform = transform
        self.size = 0
        self.imgs_list = []

        file = open(self.imgs_path)  # 我们使⽤open()函数来打开⼀个⽂件, 获取到⽂件句柄.
        for f in file:
            self.imgs_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.imgs_list[idx].split(' ')[0]  # split(' ')按照空格分隔imgs_list [n]:表示选取第n个分片
        image = Image.open(image_path).convert('RGB')  # 读取图片
        if(image.mode == 'L'):
             image = image.convert('RGB')
        label = int(self.imgs_list[idx].split(' ')[1])
        #image = np.array(image)
        label = np.array(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label