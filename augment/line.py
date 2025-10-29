import cv2
import numpy as np
from PIL import Image

class SobelEdgeDetector(object):
    def __init__(self, dx=1, dy=0, ksize=3):
        """
        参数:
            dx: x方向导数阶数 (0或1)
            dy: y方向导数阶数 (0或1)
            ksize: Sobel核大小 (1,3,5,7)
        """
        if dx not in [0, 1] or dy not in [0, 1]:
            raise ValueError("dx和dy必须为0或1")
        if ksize not in [1, 3, 5, 7]:
            raise ValueError("ksize 必须是 1, 3, 5 或 7")
        self.dx = dx
        self.dy = dy
        self.ksize = ksize
 
    def __call__(self, img):
        """
        执行Sobel边缘检测
        参数:
            img: 输入图像 (PIL.Image 或 numpy.ndarray)
        返回:
            edge_img: 边缘检测结果图像 (numpy.ndarray)
        """
        if isinstance(img, Image.Image):
            img = np.array(img.convert('RGB'))  # 转换为灰度图像的numpy数组
        elif not isinstance(img, np.ndarray):
            raise TypeError(f"Expected img to be a PIL.Image or numpy.ndarray, but got {type(img)}")
 
        sobel_x = cv2.Sobel(img, cv2.CV_64F, self.dx, self.dy, ksize=self.ksize)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, self.dy, self.dx, ksize=self.ksize)
        edge_img = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
  
        return edge_img
 
# 包装 SobelEdgeDetector 以返回 PIL.Image（如果需要）
class SobelEdgeDetectorTransform(object):
    def __init__(self, dx=1, dy=0, ksize=3):
        self.sobel_detector = SobelEdgeDetector(dx, dy, ksize)
 
    def __call__(self, img):
        edge_img_np = self.sobel_detector(img)
        # 如果需要返回 PIL.Image
        return Image.fromarray(edge_img_np)

class CannyEdgeDetector(object):
    def __init__(self, threshold1=100, threshold2=200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
 
    def __call__(self, img):
        # 检查输入类型
        if isinstance(img, str):  # 如果是字符串路径（这里不适用，因为 Compose 不会传字符串）
            raise ValueError("CannyEdgeDetector does not support string paths in Compose")
        elif isinstance(img, Image.Image):
            # 如果是 PIL.Image 对象，先转换为 numpy 数组
            img = np.array(img)
            # 检查是否为灰度图像，如果不是则转换为灰度图像
            if len(img.shape) == 3:  # 彩色图像 (H, W, C)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif not isinstance(img, np.ndarray):
            raise TypeError(f"Expected img to be a PIL.Image or numpy.ndarray, but got {type(img)}")
        else:
            # 如果是 numpy 数组，检查是否为灰度图像
            if len(img.shape) == 3:  # 彩色图像 (H, W, C)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif len(img.shape) != 2:  # 不是灰度图像也不是彩色图像
                raise ValueError("Input image must be a 2D grayscale image or 3D RGB image")
 
        # 应用 Canny 边缘检测
        edge_img = cv2.Canny(img, self.threshold1, self.threshold2)
        
        # 将单通道灰度图像转换为 3 通道图像
        edge_img_3_channel = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        
        # 转换回 PIL.Image
        return Image.fromarray(edge_img_3_channel)

class LaplacianEdgeDetector:
    def __init__(self, ddepth=cv2.CV_64F, ksize=1):
        """
        参数:
            ddepth: 输出图像的深度 (如 cv2.CV_64F)
            ksize: Laplacian算子的核大小 (1, 3, 5, 7)，必须为奇数
        """
        if ksize not in [1, 3, 5, 7]:
            raise ValueError("ksize 必须是 1, 3, 5 或 7")
        self.ddepth = ddepth
        self.ksize = ksize
 
    def __call__(self, img):
        """
        执行Laplacian边缘检测
        参数:
            img: 输入图像 (PIL.Image 或 numpy.ndarray)
        返回:
            edge_img: 边缘检测结果图像 (PIL.Image)
        """
        # 如果输入是PIL.Image，则转换为numpy数组
        if isinstance(img, Image.Image):
            img = np.array(img.convert('RGB'))  # 转换为灰度图像的numpy数组
        elif not isinstance(img, np.ndarray):
            raise TypeError(f"Expected img to be a PIL.Image or numpy.ndarray, but got {type(img)}")
 
        # 执行Laplacian边缘检测
        edge_img = cv2.Laplacian(img, self.ddepth, ksize=self.ksize)
        edge_img = np.uint8(np.absolute(edge_img))  # 转换为无符号8位整数
 
        # 转换回PIL.Image
        return Image.fromarray(edge_img)

class LoGEdgeDetector(object):
    def __init__(self, kernel_size=5):
        """
        参数:
            kernel_size: 高斯核大小（必须是正奇数）
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是正奇数")
        self.kernel_size = kernel_size
 
    def __call__(self, img):
        """
        执行LoG边缘检测
        参数:
            img: 输入图像 (PIL.Image)
        返回:
            edge_img: 边缘检测结果图像 (PIL.Image)，三通道
        """
        # 将 PIL.Image 转换为 numpy.ndarray
        img_np = np.array(img)
 
        # 确保图像是灰度图，如果不是，则转换为灰度图（虽然这里假设输入可以是彩色，但为了统一处理，还是进行转换）
        if len(img_np.shape) == 3:  # 彩色图像 (H, W, C)
            img_np_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_np_gray = img_np  # 已经是灰度图
 
        # 应用高斯模糊
        img_blurred = cv2.GaussianBlur(img_np_gray, (self.kernel_size, self.kernel_size), 0)
 
        # 执行拉普拉斯边缘检测
        edge_np = cv2.Laplacian(img_blurred, cv2.CV_64F)
        edge_np = np.uint8(np.absolute(edge_np))
 
        # 将单通道边缘图像转换为三通道
        edge_np_3channel = cv2.cvtColor(edge_np, cv2.COLOR_GRAY2RGB)  # 或者使用 np.stack((edge_np,)*3, axis=-1)
 
        # 将结果转换回 PIL.Image
        edge_img = Image.fromarray(edge_np_3channel)
        return edge_img

class PrewittEdgeDetector(object):
    def __init__(self):
        # 初始化 Prewitt 核
        self.kernel_x = np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]], dtype=np.float32)
        self.kernel_y = np.array([[-1, -1, -1],
                                  [ 0,  0,  0],
                                  [ 1,  1,  1]], dtype=np.float32)
 
    def __call__(self, img):
        """
        执行 Prewitt 边缘检测
        参数:
            img: 输入图像，可以是 numpy 数组或 PIL 图像
        返回:
            edge_img: 边缘检测结果图像 (numpy 数组)
        """
        # 如果输入是 PIL 图像，则先转换为 numpy 数组
        if hasattr(img, 'convert'):  # 检查是否为 PIL 图像
            img = np.array(img.convert('RGB'))  # 转换为灰度图
        elif not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a PIL Image or numpy array.")
        
        # 应用 Prewitt 核
        x_grad = cv2.filter2D(img, -1, self.kernel_x)
        y_grad = cv2.filter2D(img, -1, self.kernel_y)
        
        # 计算梯度幅值
        edge_img = np.sqrt(x_grad**2 + y_grad**2).astype(np.uint8)
        return edge_img
    
class ScharrEdgeDetector(object):
    def __init__(self, threshold=128):
        """
        初始化 ScharrEdgeDetector
        参数:
            threshold: 二值化的阈值，默认值为 128
        """
        self.threshold = threshold
 
    def __call__(self, img):
        """
        执行 Scharr 边缘检测并二值化，输出为 RGB 图像
        参数:
            img: 输入图像，可以是 numpy 数组或 PIL 图像
        返回:
            scharr_edge_img_rgb: 二值化 Scharr 边缘检测结果图像 (RGB 格式的 numpy 数组)
        """
        # 如果输入是 PIL 图像，则先转换为 numpy 数组
        if hasattr(img, 'convert'):  # 检查是否为 PIL 图像
            img = np.array(img.convert('L'))  # 转换为灰度图
        elif not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a PIL Image or numpy array.")
        
        # 使用 Scharr 算子计算 x 和 y 方向的梯度
        scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        
        # 计算梯度幅值
        scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2).astype(np.uint8)
        
        # 二值化
        _, scharr_edge_img = cv2.threshold(scharr_magnitude, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 转换为 RGB 格式
        scharr_edge_img_rgb = cv2.cvtColor(scharr_edge_img, cv2.COLOR_GRAY2RGB)
        
        return scharr_edge_img_rgb
class RobertsEdgeDetector(object):
    def __init__(self, threshold=128):
        """
        初始化 RobertsEdgeDetector
        参数:
            threshold: 二值化的阈值，默认值为 128
        """
        self.threshold = threshold
        # 定义 Roberts 核
        self.kernel_x = np.array([[1, 0],
                                  [0, -1]], dtype=np.float32)
        self.kernel_y = np.array([[0, 1],
                                  [-1, 0]], dtype=np.float32)
 
    def __call__(self, img):
        """
        执行 Roberts 边缘检测并二值化，输出为 RGB 图像
        参数:
            img: 输入图像，可以是 numpy 数组或 PIL 图像
        返回:
            roberts_edge_img_rgb: 二值化 Roberts 边缘检测结果图像 (RGB 格式的 numpy 数组)
        """
        # 如果输入是 PIL 图像，则先转换为 numpy 数组
        if hasattr(img, 'convert'):  # 检查是否为 PIL 图像
            img = np.array(img.convert('L'))  # 转换为灰度图
        elif not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a PIL Image or numpy array.")
        
        # 应用 Roberts 核
        x_grad = cv2.filter2D(img, -1, self.kernel_x)
        y_grad = cv2.filter2D(img, -1, self.kernel_y)
        
        # 计算梯度幅值
        edge_img = np.sqrt(x_grad**2 + y_grad**2).astype(np.uint8)
        
        # 二值化
        _, roberts_edge_img = cv2.threshold(edge_img, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 转换为 RGB 格式
        roberts_edge_img_rgb = cv2.cvtColor(roberts_edge_img, cv2.COLOR_GRAY2RGB)
        
        return roberts_edge_img_rgb
class DOGEdgeDetector(object):
    def __init__(self, kernel_size1=5, sigma1=1.0, kernel_size2=5, sigma2=2.0):
        """
        Difference of Gaussians (DoG) Edge Detector
        
        Args:
            kernel_size1 (int): Kernel size for first Gaussian blur (must be odd)
            sigma1 (float): Sigma for first Gaussian blur
            kernel_size2 (int): Kernel size for second Gaussian blur (must be odd)
            sigma2 (float): Sigma for second Gaussian blur
        """
        if kernel_size1 % 2 == 0 or kernel_size2 % 2 == 0:
            raise ValueError("kernel_size1 and kernel_size2 must be odd numbers")
        self.kernel_size1 = kernel_size1
        self.sigma1 = sigma1
        self.kernel_size2 = kernel_size2
        self.sigma2 = sigma2
        
    def __call__(self, img):
        """
        执行DOG边缘检测
        参数:
            img: 输入图像 (PIL.Image)
        返回:
            edge_img: 边缘检测结果图像 (PIL.Image)，三通道
        """
        # 将 PIL.Image 转换为 numpy.ndarray
        img_np = np.array(img)
        
        # 确保图像是灰度图，如果不是，则转换为灰度图
        if len(img_np.shape) == 3:  # 彩色图像 (H, W, C)
            img_np_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_np_gray = img_np  # 已经是灰度图
        
        # 应用第一个高斯模糊
        img_blurred1 = cv2.GaussianBlur(img_np_gray, (self.kernel_size1, self.kernel_size1), self.sigma1)
        
        # 应用第二个高斯模糊
        img_blurred2 = cv2.GaussianBlur(img_np_gray, (self.kernel_size2, self.kernel_size2), self.sigma2)
        
        # 计算DoG (Difference of Gaussians)
        dog = img_blurred1 - img_blurred2
        
        # 取绝对值并归一化到[0, 255]
        edge_np = np.uint8(np.absolute(dog))
        edge_np = cv2.normalize(edge_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # 将单通道边缘图像转换为三通道
        edge_np_3channel = cv2.cvtColor(edge_np, cv2.COLOR_GRAY2RGB)
        
        # 将结果转换回 PIL.Image
        edge_img = Image.fromarray(edge_np_3channel)
        return edge_img
