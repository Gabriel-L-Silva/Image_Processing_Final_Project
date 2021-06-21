# Nome: Diogo Godoi e Gabriel Lucas da Silva
# NUSP: 11936471 e 12620283
# Pós Graduação - SCC5830
# 1 semestre
# Processamento de imagens
# Final project

import numpy as np
import cv2
import imageio
from numpy.lib.function_base import copy
from skimage import morphology
import matplotlib.pyplot as plt
import os
import copy
from tqdm import tqdm

PATH = "./videos/"

def haar_cascade(image):
    img = copy.deepcopy(image)
    # implementation from https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
    
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    gray = img
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return img

def skin_detection(img):
    # implementation from https://github.com/CHEREF-Mehdi/SkinDetection
    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)


    #show results
    cv2.imshow("1_HSV.jpg",HSV_result)
    cv2.imshow("2_YCbCr.jpg",YCrCb_result)
    cv2.imshow("3_global_result.jpg",global_result)
    cv2.imshow("Image.jpg",img)
    cv2.imwrite("1_HSV.jpg",HSV_result)
    cv2.imwrite("2_YCbCr.jpg",YCrCb_result)
    cv2.imwrite("3_global_result.jpg",global_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_bg(videopath):
    backSub = cv2.createBackgroundSubtractorKNN()

    capture = cv2.VideoCapture(videopath)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        morph = morphology.disk(9)
        fgMask = backSub.apply(frame)
        th, fgMask = cv2.threshold(fgMask,100,255,cv2.THRESH_BINARY)
        fgMask = morphology.closing(fgMask, morph)
        
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)
        
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def bg_medium(frames):
    # implementation from https://towardsdatascience.com/background-removal-with-python-b61671d1508a
    nobg_frames = []
    for frame in tqdm(frames):
        # Parameters
        blur = 21
        canny_low = 15
        canny_high = 150
        min_area = 0.0005
        max_area = 0.95
        mask_dilate_iter = 10
        mask_erode_iter = 10
        mask_color = (0.0,0.0,0.0)
        # Convert image to grayscale      
        image_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)        
        # Apply Canny Edge Dection
        edges = cv2.Canny(image_gray, canny_low, canny_high)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
        # Get the area of the image as a comparison
        image_area = frame.shape[0] * frame.shape[1]  
        
        # calculate max and min areas in terms of pixels
        max_area = max_area * image_area
        min_area = min_area * image_area
        # Set up mask with a matrix of 0's
        mask = np.zeros(edges.shape, dtype = np.uint8)
        # Go through and find relevant contours and apply to mask
        for contour in contour_info:            
            # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
            if contour[1] > min_area and contour[1] < max_area:
                # Add contour to mask
                mask = cv2.fillConvexPoly(mask, contour[0], (255))
        # use dilate, erode, and blur to smooth out the mask
        mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
        mask = cv2.erode(mask, None, iterations=mask_erode_iter)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        # Ensures data types match up
        mask_stack = np.dstack([mask]*3)
        mask_stack = mask_stack.astype('float32') / 255.0           
        frame = frame.astype('float32') / 255.0
        # Blend the image and the mask
        masked = (mask_stack * frame) + ((1-mask_stack) * mask_color)
        masked = (masked * 255).astype('uint8')
        nobg_frames.append(masked)
    # for i, frame in enumerate(nobg_frames):
    #     cv2.imshow('Frame', frame)
    #     cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #     cv2.putText(frame, str(i), (15, 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    #     cv2.waitKey(0)
    return np.asarray(nobg_frames)
        
def morph_disk(img):
    img_erosion_disk3 = morphology.erosion(img, morphology.square(9)).astype(np.uint8)   
    img_delation_disk3 = morphology.dilation(img, morphology.square(9)).astype(np.uint8)    
    img_grad = img_delation_disk3 - img_erosion_disk3
    # plt.subplot(141); plt.imshow(img, cmap='gray')
    # plt.subplot(142); plt.imshow(img_erosion_disk3, cmap='gray')
    # plt.subplot(143); plt.imshow(img_delation_disk3, cmap='gray')
    # plt.subplot(144); plt.imshow(img_grad, cmap='gray')
    # plt.show()

    for i in range(0,100,5):
        ret, th_adpt = cv2.threshold(img,220-i,255,cv2.THRESH_BINARY)

        cv2.imshow('threshold grad',th_adpt)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    circles = cv2.HoughCircles(img_erosion_disk3,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img_erosion_disk3,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img_erosion_disk3,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',img_erosion_disk3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_erosion_disk3

def main():
    for filename in os.listdir(PATH):
        if filename.endswith('.npy') or filename.startswith('output'):
            continue
        
        # Opens the Video file
        # cap= cv2.VideoCapture(PATH+filename)
        # frames = []
        # while(cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret == False:
        #         break
        #     frames.append(frame)
        # cap.release()
        # cv2.destroyAllWindows()

        # frames = np.asarray(frames)
        
        # nobg_frames = bg_medium(frames)
        # remove_bg(PATH+filename)
        # for frame in frames:
        #     skin_detection(img)

        #     haar = haar_cascade(img)
        #     morph_img = morph_disk(img)
            
            # plt.subplot(131)
            # plt.imshow(haar, cmap='gray')
            # plt.subplot(132)
            # plt.imshow(morph_img, cmap='gray')
            # plt.subplot(133)
            # plt.imshow(img-morph_img, cmap='gray')
            # plt.show()
        # np.save(PATH+filename, nobg_frames, allow_pickle=True)
        nobg_frames = np.load(PATH+filename+'.npy',allow_pickle=True)
        out = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (720,1080))
        nobg_frames = nobg_frames.reshape(nobg_frames.shape[0],1280,720,3)
        for frame in tqdm(nobg_frames):
            out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)
        out.release()
        
    return
    
if __name__ == '__main__':
    main()