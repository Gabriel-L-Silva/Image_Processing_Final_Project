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
from skimage import morphology
import os
import copy
from tqdm import tqdm
# import matplotlib.pyplot as plt

PATH = "./videos/"


#Usado para comparar com o nosso método
def haar_cascade(img):
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

def skin_detection(imgs):
    ''' Skin detection
    implementation from: https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/

    After skin is detected, we use it as a mask to remove it from the images
    '''
    masked_imgs = []
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    print('\n'+"skin detection")
    for frame in tqdm(imgs):
        # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the speicifed upper and lower boundaries
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)
        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.bitwise_not(cv2.GaussianBlur(skinMask, (3, 3), 0))
        
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)
        masked_imgs.append(skin)
        # show the skin in the image along with the mask
        # cv2.imshow("images", skinMask)
        # if the 'q' key is pressed, stop the loop
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return masked_imgs

def remove_bg(frames, frame_jump=0):
    ''' Remove backgroung using basic open cv example, jumps every frame_jump. 
    Uses opencv contour to find biggest contour and fill it on the mask, then 
    uses the mean of all frames processed to remove BG. 

    paramaters:
        frames: set of images to remove background
        frame_jump: number of frames jumped
    '''
    backSub = cv2.createBackgroundSubtractorKNN()
    # backSub = cv.createBackgroundSubtractorMOG2()
    masks = []
    nobg_frames = []
    for n_f, frame in enumerate(tqdm(frames)):
        if (n_f+1)%frame_jump != 0:
            continue
        fgMask = backSub.apply(frame)
        # Threshold to remove shadows(gray color)
        th, fgMask = cv2.threshold(fgMask,150,255,cv2.THRESH_BINARY)
        # morph = morphology.disk(9)
        # fgMask = morphology.closing(fgMask, morph)
        
        masks.append(fgMask)
    _, mask = cv2.threshold(np.mean(np.asarray(masks),axis=0),100,255,cv2.THRESH_BINARY)
    mask = mask.astype('uint8')
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # Find biggesdt contour and fill it with white color
    # https://www.programmersought.com/article/49006381746/
    c_max = []  
    max_area = 0  
    max_cnt = 0  
    for i in range(len(contours)):  
        cnt = contours[i]  
        area = cv2.contourArea(cnt)  
        # find max countour  
        if (area>max_area):  
            if(max_area!=0):  
                c_min = []  
                c_min.append(max_cnt)  
                cv2.drawContours(mask, c_min, -1, (255,255,255), cv2.FILLED)  
            max_area = area  
            max_cnt = cnt  
        else:  
            c_min = []  
            c_min.append(cnt)  
            cv2.drawContours(mask, c_min, -1, (255,255,255), cv2.FILLED)  
    c_max.append(max_cnt)

    # Apply average mask to frames
    for frame in tqdm(frames):
        # Blend the image and the mask
        masked = cv2.bitwise_and(frame,np.dstack([mask]*3))
        # masked = (masked * 255).astype('uint8')
        nobg_frames.append(masked)
    # for i, frame in enumerate(nobg_frames):
    #     cv2.imshow('Frame', frame)
    #     cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #     cv2.putText(frame, str(i), (15, 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return nobg_frames

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def detect_blobs(img, params):
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Accept as eyes only the blobs within same vertical range and mininum 
    # horizontal distance
    blobs = []
    min_dist = np.inf
    for A in keypoints:
        for B in keypoints:
            if A != B:
                maxA = A.pt[1] + A.size//2
                minA = A.pt[1] - A.size//2
                maxB = B.pt[1] + B.size//2
                minB = B.pt[1] - B.size//2
                if not(maxA > minB and maxB > minA):
                    continue
                dist = distance(A.pt,B.pt)
                if dist < min_dist:
                    min_dist = dist
                    blobs = [A,B]

    return blobs, len(keypoints)

def morph_disk(img, params):
    ''' Tries to detect the eyes from a image without background and skin

    Uses morphology to detect the eyes sharp point of reflection(light dot), we 
    used the algorithm suggested by Rajpathak, Tanmay & Kumar, Ratnesh & Schwartz, Eric. (2009). Eye Detection Using Morphological and Color Image Processing
    
    We added some changes that should enhance our resutls
    -parameters:
        img: image to aplly the algorithm
        params: parameters for blobs detection from open cv
    '''
    img_erosion = morphology.erosion(img, morphology.square(9)).astype(np.uint8) 
    img_delation = morphology.dilation(img, morphology.square(9)).astype(np.uint8)    
    img_grad = img_delation - img_erosion
    
    # Aplies an adaptative threshold that aims to have 4 to 6 blobs, with not 
    # the value of threshold is lowered
    for i in range(0,40,5):
        ret, th_adpt_closed = cv2.threshold(img_grad,220-i,255,cv2.THRESH_BINARY)
        blobs, n_keypoints = detect_blobs(th_adpt_closed, params)
        # cv2.imshow('threshold',th_adpt_closed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        if 4 <= n_keypoints <= 6:
            # Remove small pepper noise
            th_adpt_closed = morphology.binary_erosion(th_adpt_closed,morphology.square(3))
            # Tries to connect element from the same eye horizontally
            th_adpt_closed = morphology.binary_dilation(th_adpt_closed, morphology.rectangle(10,70))
            # Close small dark gaps to prevent from detecting false pair of eye
            th_adpt_closed = morphology.binary_closing(th_adpt_closed, morphology.square(13)).astype('uint8')*255
            blobs, n_keypoints = detect_blobs(th_adpt_closed, params)
            return blobs

def main():
    #Load video into list of frames
    for filename in os.listdir(PATH):
        if filename.endswith('.npy') or filename.endswith('.avi') or os.path.isdir(PATH+filename):
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
              
        # Remove background using open cv example
        nobg_frames = remove_bg(frames, 25)
        
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 255
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByColor = False
        params.filterByCircularity = False

        # Remove skin from the frames without background
        masked_frames = skin_detection(nobg_frames)
        # for img in masked_frames:
        #     cv2.imshow('Eye tracker',img)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #Applies the suggests algorithm and draw a rectangle on the eyes found
        result_frames = copy.deepcopy(frames)
        for i, frame in enumerate(tqdm(nobg_frames)):
            eye_pos = morph_disk(cv2.cvtColor(masked_frames[i], cv2.COLOR_RGB2GRAY), params)
            if eye_pos == None:
                print(f"no eye found on frame {i}")
            else:
                for eye in eye_pos:
                    tl = (int(eye.pt[0]-eye.size//2),int(eye.pt[1]+eye.size//2))
                    rb = (int(eye.pt[0]+eye.size//2),int(eye.pt[1]-eye.size//2))
                    cv2.rectangle(result_frames[i], tl, rb, (255,0,0))
                # cv2.imshow('Eye tracker',frames[i])
                # cv2.waitKey(0)                
        # cv2.destroyAllWindows()
        
        # Save the result in a video
        out = cv2.VideoWriter(PATH+"eyes_"+filename[:-4]+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280,720))
        for frame in result_frames:
            # writing to a image array
            out.write(frame)
        out.release()

        # Apply haar cascades algorithm and saves it on video for comparison
        haar_frames = copy.deepcopy(frames)        
        out = cv2.VideoWriter(PATH+"haar_"+filename[:-4]+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280,720))
        for frame in tqdm(haar_frames):
            # writing to a image array
            haar_cascade(frame)
            out.write(frame)
        out.release()

    return
    
if __name__ == '__main__':
    # profile = cProfile.Profile()
    # profile.runcall(main)
    # ps = pstats.Stats(profile)
    # ps.print_stats()
    main()