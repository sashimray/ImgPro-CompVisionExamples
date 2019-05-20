## Project 2 
# Gaussian Blur 
# Unsharp Masking
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def unsharp_masking(img, blur):
    sharped_image = cv.addWeighted(img, 1.5, blur, -0.5, 0, img)
    
    plt.hist(img.ravel(),256,[0,256]);
    plt.show()
    plt.hist(sharped_image.ravel(),256,[0,256]);
    plt.show()
    
    cv.imshow('og',img)
    cv.imshow('blurred',blur)
    cv.imshow('unsharp_masking',sharped_image)

    cv.waitKey(0)
    cv.destroyAllWindows()
    
if __name__== "__main__": 
    image = cv.imread('p2.jpg', 0)
    
    #Source :https://www.idtools.com.au/unsharp-masking-python-opencv/
    blur_9 = cv.GaussianBlur(image,(9,9),10.0) 
    blur_11 = cv.GaussianBlur(image,(5,5), 10.0)
    
    #plot histogram for the original and blurred images 
    #Source : https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
    plt.hist(image.ravel(),256,[0,256]);
    plt.hist(blur_9.ravel(),256,[0,256]);
    plt.hist(blur_11.ravel(),256,[0,256]);
    plt.show()
    
    # Unsharp Masking with blur_3
    # Source : https://stackoverflow.com/questions/32454613/python-unsharp-mask
    unsharp_masking(image, blur_9)
    image2 = cv.imread('p1.jpg')
    blur_bgr = cv.GaussianBlur(image2 ,(9,9),10.0) 
    unsharp_masking(image2, blur_bgr)
    
    