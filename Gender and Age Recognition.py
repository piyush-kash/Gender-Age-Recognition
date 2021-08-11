#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
os.chdir('D:\My Projects\Gender & Age Recognition\models')


# In[6]:


#Function to detect Face

def faceDetect(net,frame,confidence_threshold=0.7):    
    frameOpencvDNN=frame.copy()           
    frameHeight=frameOpencvDNN.shape[0]                      
    frameWidth=frameOpencvDNN.shape[1]                      
    blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False) 
    detections=net.forward()             
    
    faceBoxes=[]                          
    
    for i in range(detections.shape[2]):     
        confidence=detections[0,0,i,2]      
        if confidence > confidence_threshold:          
            x1=int(detections[0,0,i,3]*frameWidth)     
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])            
            cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)   
                                                          
    return frameOpencvDNN,faceBoxes        


#Loading the Pre-trained Caffe models

faceProto='opencv_face_detector.pbtxt'
faceModel='opencv_face_detector_uint8.pb'
ageProto='age_deploy.prototxt'
ageModel='age_net.caffemodel'
genderProto='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'

genderList=['Male','Female']
ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']   

faceNet=cv2.dnn.readNet(faceModel,faceProto)          
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


video=cv2.VideoCapture(0)                  

while cv2.waitKey(1)<0:                   
    hasFrame,frame=video.read()           
    if not hasFrame:                      
        cv2.waitKey()
        break
    resultImg,faceBoxes=faceDetect(faceNet,frame)   
    
    if not faceBoxes:   
        print("No face detected")
        
    for faceBox in faceBoxes:                         
        blob=cv2.dnn.blobFromImage(resultImg,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)    
        genderNet.setInput(blob)            
        genderPreds=genderNet.forward()      
        gender=genderList[genderPreds[0].argmax()]  
        
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]   
        cv2.putText(resultImg, f'{gender},{age}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,(0,255,255),2,cv2.LINE_AA) 
        cv2.imshow("Detecting Age & Gender now..",resultImg)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):    
            break

            
cv2.destroyAllWindows()


# In[ ]:




