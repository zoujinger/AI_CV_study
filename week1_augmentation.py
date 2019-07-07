# combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


# image crop
def image_crop(img):
    h, w, ch = img.shape
    h1 = random.randint(0, h)
    w1 = random.randint(0, w)
    if h1 < h/2 and w1 < w/2:
        return img[h1:h, w1:w]
    elif h1 < h/2 and w1 > w/2:
        return img[h1:h, 0:w]
    elif h1 > h/2 and w1 < w/2:
        return img[0:h1, w1:w]
    elif h1 > h / 2 and w1 > w / 2:
        return img[0:h, 0:w]


# color shift
def random_light_color(img):
    B, G, R=cv2.split(img)
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand <0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    img_merge = cv2.merge((B, G, R))
    return img_merge


# rotation
def image_rotation(img, scale=1):
    angle = random.randint(0, 360)
    m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, scale)  # center angle scale
    img_rotate = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
    return img_rotate


# perspective transform
def perspective_transform(img):
    height, width, channels = img.shape

    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    m_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, m_warp, (width, height))
    return img_warp


if __name__ == '__main__':
    img_origin = cv2.imread('/Users/jinger/Downloads/lenna.png')
    img1 = image_crop(img_origin)
    img2 = random_light_color(img_origin)
    img3 = image_rotation(img_origin)
    img4 = perspective_transform(img_origin)

    imgs = np.hstack([img2, img3, img4])
    # 展示多个
    cv2.imshow("mutil_pic", imgs)

    # plt.subplot(1, 4, 1)
    cv2.imshow('croped', img1)
    
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()