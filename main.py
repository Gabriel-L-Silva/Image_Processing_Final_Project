# Nome: Diogo Godoi e Gabriel Lucas da Silva
# NUSP: 11936471 e 12620283
# Pós Graduação - SCC5830
# 1 semestre
# Processamento de imagens
# Final project

import cProfile
import pstats
import numpy as np
import cv2
from numpy.core.numeric import zeros_like
from skimage import morphology
import os
import copy
from tqdm import tqdm

PATH = "./videos/"

def haar_cascade(image):
    img = copy.deepcopy(image)
    # implementation from https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
    
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haar_configs/haarcascade_frontalface_default.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haar_configs/haarcascade_eye.xml')

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


    masked_img = cv2.bitwise_and(img,np.dstack([global_result]*3))
    #show results
    # cv2.imshow("1_HSV.jpg",HSV_result)
    # cv2.imshow("2_YCbCr.jpg",YCrCb_result)
    # cv2.imshow("3_global_result.jpg",global_result)
    # cv2.imshow("Image.jpg",img)
    # cv2.imshow("Skinless image.jpg", masked_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return masked_img

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

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def morph_disk(img):
    img_erosion_disk3 = morphology.erosion(img, morphology.square(9)).astype(np.uint8) 
    img_delation_disk3 = morphology.dilation(img, morphology.square(9)).astype(np.uint8)    
    img_grad = img_delation_disk3 - img_erosion_disk3
    # plt.subplot(141); plt.imshow(img, cmap='gray')
    # plt.subplot(142); plt.imshow(img_erosion_disk3, cmap='gray')
    # plt.subplot(143); plt.imshow(img_delation_disk3, cmap='gray')
    # plt.imshow(img_grad, cmap='gray')
    # plt.show()

    # for i in range(0,100,5):
    ret, th_adpt = cv2.threshold(img_grad,220,255,cv2.THRESH_BINARY)
    th_adpt_closed = morphology.closing(th_adpt, morphology.disk(25))
    th_adpt_closed = morphology.dilation(th_adpt_closed, morphology.rectangle(10,30))
    th_adpt_closed = morphology.closing(th_adpt_closed, morphology.square(10))
    # cv2.imshow('threshold grad',th_adpt_closed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 255

    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False
    params.filterByCircularity = False
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(th_adpt_closed)

    blobs = []
    min_dist = np.inf
    for A in keypoints:
        for B in keypoints:
            if A != B:
                if A.pt[1] > B.pt[1] and B.pt[1] < A.pt[1]:
                    continue
                dist = distance(A.pt,B.pt)
                if dist < min_dist:
                    min_dist = dist
                    blobs = [A,B]
                


    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv2.drawKeypoints(th_adpt_closed, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blobs

def main():
    for filename in os.listdir(PATH):
        if filename.endswith('.npy') or filename.endswith('.avi'):
            continue
        # Opens the Video file
        cap= cv2.VideoCapture(PATH+filename)
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frames.append(frame)
        cap.release()
        cv2.destroyAllWindows()

        frames = np.asarray(frames)
        if filename[:-4]+".npy" in os.listdir(PATH):
            nobg_frames = np.load(PATH+filename[:-4]+'.npy',allow_pickle=True)
        else:            
            nobg_frames = bg_medium(frames)
            # remove_bg(PATH+filename)
            # for frame in frames:
            #     skin_detection(frame)

            #     haar = haar_cascade(frame)
            #     morph_img = morph_disk(frame)
                
                # plt.subplot(131)
                # plt.imshow(haar, cmap='gray')
                # plt.subplot(132)
                # plt.imshow(morph_img, cmap='gray')
                # plt.subplot(133)
                # plt.imshow(frame-morph_img, cmap='gray')
                # plt.show()
            np.save(PATH+filename[:-4], nobg_frames, allow_pickle=True)
        if "nobg_"+filename[:-4]+".avi" not in os.listdir(PATH):
            out = cv2.VideoWriter(PATH+"nobg_"+filename[:-4]+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280,720))
            for frame in nobg_frames:
                # writing to a image array
                out.write(frame)
            out.release()
        
        for i, frame in enumerate(tqdm(nobg_frames)):
            masked_frame = skin_detection(frame)
            eye_pos = morph_disk(cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY))
            for eye in eye_pos:
                tl = (int(eye.pt[0]-eye.size//2),int(eye.pt[1]+eye.size//2))
                rb = (int(eye.pt[0]+eye.size//2),int(eye.pt[1]-eye.size//2))
                cv2.rectangle(frames[i], tl, rb, (255,0,0))
            # cv2.imshow('Eye tracker',frames[i])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
    
        out = cv2.VideoWriter(PATH+"eyes_"+filename[:-4]+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280,720))
        for frame in frames:
            # writing to a image array
            out.write(frame)
        out.release()
    return
    
if __name__ == '__main__':
    profile = cProfile.Profile()
    profile.runcall(main)
    ps = pstats.Stats(profile)
    ps.print_stats()