# sources : 
# https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
# https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_image_histogram_calcHist.php
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

test_img = cv.imread('somethingBlue.jpg')
img1 = cv.imread('exam1.jpg')  ##indoors
img2 = cv.imread('img_out.jpg')  ## outdoors
img3 = cv.imread('img_outsh.jpg') ## outdoor shaded 
  
def convert_image():
    ## first assignment 
    
    hsv = cv.cvtColor(test_img, cv.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(test_img, test_img, mask= mask)
    
    lab = cv.cvtColor(hsv, cv.COLOR_BGR2LAB)
    #show all images 
    cv.imshow('frame',test_img)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    cv.imshow('lab', lab)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def create_histogram():

    #calculate hist for image
### cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    
    ##bin size = 8
    hist1= cv.calcHist([img1],[0],None, [8],[0,256])
    plt.plot(hist1)
#    hist2= cv.calcHist([img2],[0],None, [8],[0,256])
#    plt.plot(hist2)
#    hist3= cv.calcHist([img3],[0],None, [8],[0,256])
#    plt.plot(hist3)
    plt.show()
    
#    ##bin size = 16
    hist1= cv.calcHist([img1],[0],None, [16],[0,256])
    plt.plot(hist1)
#    hist2= cv.calcHist([img2],[0],None, [16],[0,256])
#    plt.plot(hist2)
#    hist3= cv.calcHist([img3],[0],None, [16],[0,256])
#    plt.plot(hist3)
    plt.show()
#    
#    ##bin size = 256
    hist1= cv.calcHist([img1],[0],None, [256],[0,256])
    plt.plot(hist1)
#    hist2= cv.calcHist([img2],[0],None, [256],[0,256])
#    plt.plot(hist2)
#    hist3= cv.calcHist([img3],[0],None, [256],[0,256])
#    plt.plot(hist3)
    plt.show()    
    
if __name__== "__main__": 
#    convert_image()  ## first assignment 
    create_histogram() ## second assignment 
    
    