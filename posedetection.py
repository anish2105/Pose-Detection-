import cv2
import mediapipe as mp
import numpy as np
import numpy as mnp

mPose = mp.solutions.pose
pose = mPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
drawspec1 = mpDraw.DrawingSpec(thickness = 2 , circle_radius=3 , color = (0,0,255))
drawspec2 = mpDraw.DrawingSpec(thickness = 2 , circle_radius=3 , color = (0,255,0))

while True:
    success , img = cap.read()
    img = cv2.resize(img , (800,700))
    results = pose.process(img)
    mpDraw.draw_landmarks(img, results.pose_landmarks, mPose.POSE_CONNECTIONS , drawspec1 , drawspec2)

    h,w,c = img.shape
    imgblank = np.zeros([h,w,c])
    imgblank.fill(255)
    mpDraw.draw_landmarks(imgblank, results.pose_landmarks, mPose.POSE_CONNECTIONS, drawspec1, drawspec2)

    cv2.imshow('pose' , img)
    cv2.imshow('blank', imgblank)

    cv2.waitKey(1)