import numpy as np
import cv2

def medianBlur(img, kernel, padding_way):
#        img & kernel is List of List; padding_way a string
#    What you are supposed to do can be described as "median blur", which means by using a sliding window
#    on an image, your task is not going to do a normal convolution, but to find the median value within
#    that crop.
#
#    the padding method and size. There are 2 padding ways: REPLICA & ZERO. When
#    "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version
#    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#    Follow up 1: Can it be completed in a shorter time complexity?

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', img)
    h, w = img.shape
    m, n = kernel.shape
    pad_size0 = (m-1)//2
    pad_size1 = (n-1)//2
    print('padding size:', pad_size0, pad_size1)
    # padding
    if padding_way == 'replace':
        img_pad = np.zeros((w + 2 * pad_size0, h + 2 * pad_size1), dtype=int)
        img_pad[pad_size0:pad_size0 + h, pad_size1:pad_size1 + w] = img[:]
        # pad rows
        for i in range(pad_size0):
            img_pad[i, :] = img_pad[pad_size0, :]
            img_pad[h+pad_size0+i, :] = img_pad[-pad_size0 - 1, :]
        #pad columns
        for j in range(pad_size1):
            img_pad[:, j] = img_pad[:, pad_size1]
            img_pad[:, w+pad_size1+j] = img_pad[:, -pad_size1-1]

        img_pad = img_pad.astype(np.uint8)
        print(img_pad)
    elif padding_way == 'zero':
        img_pad = np.zeros((w+2*pad_size0, h+2*pad_size1), dtype = int)
        img_pad[pad_size0:pad_size0+h, pad_size1:pad_size1+w] = img[:]
        img_pad = img_pad.astype(np.uint8)
        print(img_pad)
    #return img_pad
    cv2.imshow('img_pad', img_pad)

    # median blur
    img_blur = np.zeros((h, w), dtype=int)
    for x in range(pad_size0, img_pad.shape[0]-pad_size0-1):
        for y in range(pad_size1, img_pad.shape[1]-pad_size1-1):
            window_list = [img_pad[x-1, y-1], img_pad[x-1, y], img_pad[x-1, y+1],\
                           img_pad[x,   y-1], img_pad[x,   y], img_pad[x,   y+1],\
                           img_pad[x+1, y-1], img_pad[x+1, y], img_pad[x+1, y+1]]
            mm = median(window_list)
            img_blur[x-pad_size0, y-pad_size1] = mm

    img_blur = img_blur.astype(np.uint8)
    return img_blur


def median(alist):
    # bubble sort
    n = len(alist)
    for i in range(n):
        for j in range (n-i-1):
            if alist[j] > alist[j+1]:
                alist[j], alist[j+1] = alist[j+1], alist[j]
    med = alist[int((len(alist)-1)/2)]
    return med

if __name__ == '__main__':
    img = cv2.imread('/Users/jinger/Downloads/lenna.png')
    cv2.imshow('lenna', img)
    #kernel = np.array([[1,2,1], [0,0,0], [1,2,1]])
    kernel = np.zeros((5,5), dtype=int)
    out = medianBlur(img, kernel, 'zero')
    cv2.imshow('blurred', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
