import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def image_spectrum(img):
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return fshift
    
def inverse_fourier(img):
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    fshift[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 0
    f_ishift = np.fft.ifftshift(fshift)
    
    img_back = np.fft.ifft2(f_ishift)
    
    magnitude_spectrum = 20*np.log(np.abs(f_ishift))
    
    img_back = np.abs(img_back)
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Inverse Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show() 
    
    
def mean_filter(img):
    kernel = np.ones((5,5),np.float32)/9
    dst = cv.filter2D(img,-1,kernel)
    
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Mean')
    plt.xticks([]), plt.yticks([])
    plt.show()
    return dst

def multiplyimages(img1, img2):
    res = img1 * img2

    plt.subplot(131),plt.imshow(img1),plt.title('Image 1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img2),plt.title('Image 2')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(res),plt.title('Result')
    plt.xticks([]), plt.yticks([])

    plt.show()
    return res

def multiplyFourier(img1,img2):
    res1= image_spectrum(img1)
    res2 =image_spectrum(img2)
    res = res1 * res2
    magnitude_spectrum = 20*np.log(np.abs(res))
    plt.subplot(121),plt.imshow(magnitude_spectrum),plt.title('Fourier multiplication')
    plt.xticks([]), plt.yticks([])
    plt.show()
#    return res
    
def add_noise(img):
    noise1 = img +3 * img.std() * np.random.random(img.shape)
    alot = 4 * img.max() + np.random.random(img.shape)
    noise2 = alot + noise1
    return noise1, noise2
    
if __name__ =="__main__":
    image = cv.imread('WallStreetIn.jpg', 0)
    image_spectrum(image)    ##
    inverse_fourier(image)
    
    img1= cv.imread('test3.jpg',0)  
    image_spectrum(img1)
    dst = mean_filter(img1)
    image_spectrum (dst)
    
    img2 = cv.imread('test2.jpg',0)  
    image_spectrum(img2)
    noise_img1, noise_img2 = add_noise(img2)
    image_spectrum(noise_img1) 
    image_spectrum(noise_img2)
    
    img1= cv.imread('test3.jpg',0)
    image_spectrum(img1)
    img3 = cv.imread('test4.jpg',0) 
    image_spectrum(img3)
    
    mulImg=multiplyimages(img1,img3)
    image_spectrum(mulImg)
    
    multiplyFourier(img1,img3)