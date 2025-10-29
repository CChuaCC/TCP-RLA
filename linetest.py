#encoding:utf-8
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
 
# 读取图像（保持原始彩色）
img = cv2.imread('tcp/图片2.png')  # 默认读取为BGR格式
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式用于显示
lenna_img = img_rgb  # 使用RGB格式的彩色图像
 
# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图用于处理
 
# 高斯滤波
gaussianBlur = cv2.GaussianBlur(grayImage, (3,3), 0)
 
# 自适应阈值处理
binary = cv2.adaptiveThreshold(src=gaussianBlur, maxValue=255, 
                              adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                              thresholdType=cv2.THRESH_BINARY, blockSize=11, C=1)  
 
# Roberts算子
kernelx = np.array([[-1,0],[0,1]], dtype=int)
kernely = np.array([[0,-1],[1,0]], dtype=int)
x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
y = cv2.filter2D(binary, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)     
absY = cv2.convertScaleAbs(y)    
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
 
# Prewitt算子
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
y = cv2.filter2D(binary, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)  
absY = cv2.convertScaleAbs(y)    
Prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)
 
# Sobel算子
x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)    
absX = cv2.convertScaleAbs(x)   
absY = cv2.convertScaleAbs(y)    
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
 
# 拉普拉斯算法
dst = cv2.Laplacian(binary, cv2.CV_16S, ksize = 3)
Laplacian = cv2.convertScaleAbs(dst)
 
# Scharr算子
x = cv2.Scharr(gaussianBlur, cv2.CV_32F, 1, 0) #X方向
y = cv2.Scharr(gaussianBlur, cv2.CV_32F, 0, 1) #Y方向
absX = cv2.convertScaleAbs(x)       
absY = cv2.convertScaleAbs(y)
Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Canny算子
Canny = cv2.Canny(gaussianBlur, 20, 30)
 
# 先通过高斯滤波降噪
gaussian = cv2.GaussianBlur(grayImage, (3,3), 0)
 
# 再通过拉普拉斯算子做边缘检测
dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize = 3)
LOG = cv2.convertScaleAbs(dst)
 
# DoG算子（新增部分）
sigma1, sigma2 = 1.0, 2.0  # 高斯核参数
gaussian1 = cv2.GaussianBlur(grayImage, (0, 0), sigma1)
gaussian2 = cv2.GaussianBlur(grayImage, (0, 0), sigma2)
dog = cv2.subtract(gaussian1, gaussian2)  # DoG = G1 - G2
dog_abs = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # 归一化到[0,255]
 
# 效果图
fig = plt.figure(figsize=(12, 12))  # 设置大小
titles = ['Source Image', 'Roberts Image',
          'Prewitt Image', 'Sobel Image', 'Laplacian Image',
          'Scharr Image', 'Canny Image', 'LOG Image', 'DOG Image']
images = [lenna_img, Roberts,
          Prewitt, Sobel, Laplacian,
          Scharr, Canny, LOG, dog_abs]
 
for i in range(9):
    plt.subplot(3, 3, i + 1)
    if i == 0:
        plt.imshow(images[i])  # 彩色显示原始图像
    else:
        plt.imshow(images[i])  # 灰度显示其他结果
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
 
plt.tight_layout()
plt.show()
fig.savefig('tupian2_results.jpg', bbox_inches='tight')