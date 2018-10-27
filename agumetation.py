import numpy as np
import cv2
import random

def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# dimming
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image

#等比例缩放图片,size为最边短边长
def resize_img(img,size):
    h = img.shape[0]
    w = img.shape[1]
    if size/h>size/w:
        scale = size/h
        resized_img = cv2.resize( img, (size,int(w*scale)) )
    else:
        scale = size/w
        resized_img = cv2.resize(img, (int(h*scale), size))
    return resized_img

#对图片进行随机切割,输入图片其中一边与切割大小相等
def random_crop(img):
    h = img.shape[0]
    w = img.shape[1]
    if not w == h:
        if h>w:
            offset = random.randint(0, h - w -1)
            croped_img = img[offset:offset+w, :]
        else:
            offset = random.randint(0, w - h -1)
            croped_img = img[:, offset:offset+h]
    else:
        return img
    return croped_img

def n_fold_crop(img,n_fold=5):
    h = img.shape[0]
    w = img.shape[1]
    crop_imgs = []
    if h>w:
        intv = h - w
        stride = int(intv/(n_fold-1))
        for offset in range(0,intv,stride-1):
            crop_imgs.append(img[offset:offset+w, :])
    else:
        intv = w - h
        stride = int(intv / (n_fold-1))
        for offset in range(0, intv, stride-1):
            crop_imgs.append(img[: , offset:offset + h])
    print(intv,stride)
    return crop_imgs


def random_flip(img,chance=0.5):
    if random.random()<chance:
        img = cv2.flip(img,1)
    return img



# img = cv2.imread('F:/AgriculturalDisease/AgriculturalDisease_trainingset/images/00e6ad4a-5a62-48d7-ac68-9c0b8ec87f5f___Rut._Bact.S 1472.JPG')
# cv2.imshow("temp",img)
# cv2.waitKey()
#
# size = 224
# img = resize_img(img,size)
# cv2.imshow("temp",img)
# cv2.waitKey()
#
# crop_imgs = n_fold_crop(img)
# for crop_img in crop_imgs:
#     print(crop_img.shape)
#     cv2.imshow("temp",crop_img)
#     cv2.waitKey()
