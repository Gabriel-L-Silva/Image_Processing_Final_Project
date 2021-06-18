# Nome: Diogo Godoi e Gabriel Lucas da Silva
# NUSP: 11936471 e 12620283
# Pós Graduação - SCC5830
# 1 semestre
# Processamento de imagens
# Final project

import numpy as np
import cv2 as cv
import imageio
from numpy.lib.function_base import copy
from skimage import morphology
import matplotlib.pyplot as plt
import os
import copy

PATH = "./images/"

def haar_cascade(image):
    img = copy.deepcopy(image)
    # implementation from https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
    
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

    gray = img
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return img

def morph_disk(img):
    img_erosion_disk3 = morphology.erosion(img, morphology.disk(3)).astype(np.uint8)   
    img_delation_disk3 = morphology.dilation(img, morphology.disk(3)).astype(np.uint8)    
    img_grad = img_delation_disk3 - img_erosion_disk3
    # plt.subplot(141); plt.imshow(img, cmap='gray')
    # plt.subplot(142); plt.imshow(img_erosion_disk3, cmap='gray')
    # plt.subplot(143); plt.imshow(img_delation_disk3, cmap='gray')
    # plt.subplot(144); plt.imshow(img_grad, cmap='gray')
    # plt.show()
    
    

    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,10,
                            param1=50,param2=30,minRadius=0,maxRadius=30)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
    cv.imshow('detected circles',img)
    k = cv.waitKey(30) & 0xff
    while k != 27:
        k = cv.waitKey(30) & 0xff
    return img_erosion_disk3

def main():
    for filename in os.listdir(PATH):
        img = imageio.imread(PATH+filename)
        
        haar = haar_cascade(img)
        morph_img = morph_disk(img)
        
        # plt.subplot(131)
        # plt.imshow(haar, cmap='gray')
        # plt.subplot(132)
        # plt.imshow(morph_img, cmap='gray')
        # plt.subplot(133)
        # plt.imshow(img-morph_img, cmap='gray')
        # plt.show()
    return
    
if __name__ == '__main__':
    main()