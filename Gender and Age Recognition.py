#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
os.chdir('D:\My Projects\Gender & Age Recognition\models')


# In[6]:


#Function to detect Face

def faceDetect(net,frame,confidence_threshold=0.7):     #faceNet- the neural net #when the confidence of detecting a face is 70% or more then only it will show the result
    frameOpencvDNN=frame.copy()            #face's frame
    frameHeight=frameOpencvDNN.shape[0]                       #ex:(500,800)-- 0th index has height 1th has width
    frameWidth=frameOpencvDNN.shape[1]                        #-- to see dimensions n rgb channel3 print(frameOpencvDNN.shape)
    blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False) #blob preprocesses vdo/image to feed to cnn(we cannot give raw), 1.0 default i.e no scaling, 227.. resizing (number taken from .prototxt file),124.96 ..from website pyimage
    net.setInput(blob)                    #to face net we give blob as input
    detections=net.forward()              #process on that & give output
    
    faceBoxes=[]                          #box around face after program detects it
    
    for i in range(detections.shape[2]):     #for loop to draw that rectangle around detected face #in (1,1,116,7) [2]is 116\\ i in range(116)
        confidence=detections[0,0,i,2]       #detection will be in 4d form [[[[1,5,3]]] so to break each [] we right [0][0]...
        if confidence > confidence_threshold:           #0.7 then only show result
            x1=int(detections[0,0,i,3]*frameWidth)     #[7 csv's will be there ] coming from detections
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])            #box dimensions being appended
            cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)   #to draw rectangle around face  #255 so only green from B,G,R
                                                         #color    thickness                line type  
    return frameOpencvDNN,faceBoxes        #for every unit time box will be returned on face


#Loading the Pre-trained Caffe models

faceProto='opencv_face_detector.pbtxt'
faceModel='opencv_face_detector_uint8.pb'
ageProto='age_deploy.prototxt'
ageModel='age_net.caffemodel'
genderProto='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'

genderList=['Male','Female']
ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']   

faceNet=cv2.dnn.readNet(faceModel,faceProto)           #feeding into neural net
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


video=cv2.VideoCapture(0)                   #0 for inbulit webcam 1 is for external

while cv2.waitKey(1)<0:                    #while video is running ,until close button pressed 
    hasFrame,frame=video.read()            #frame is the input for the faceDetect fn.
    if not hasFrame:                       #hasFrame optional ..if no vdo frame break the loop
        cv2.waitKey()
        break
    resultImg,faceBoxes=faceDetect(faceNet,frame)  #fn. returned frameOpencvDNN is as resultImg 
    
    if not faceBoxes:    #it will not end program if face not found just display this message
        print("No face detected")
        
    for faceBox in faceBoxes:                         #faceBox returned as [[1,2,3,4],[4,6,7,2],[8,7,3,4]]
        blob=cv2.dnn.blobFromImage(resultImg,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)    #work on the resultImg to get a blob of 4d array n from taht no. detect age
        genderNet.setInput(blob)             #after processing give that input to genderNet
        genderPreds=genderNet.forward()       #n give output
        gender=genderList[genderPreds[0].argmax()]  #0 to break the [[]]  #from returned list of gender in output, return the max no.(showing more probality of my gender being one of them)
        #print(gender)
        
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]   #give max value of the 8 ranges returned
        #print(age)
        cv2.putText(resultImg, f'{gender},{age}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,(0,255,255),2,cv2.LINE_AA) #shift+tab at putText. to know the attributes written
                                                                #dimension
        cv2.imshow("Detecting Age & Gender now..",resultImg)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):    #waitkey --hold the button 'q'  to close the detection window
            break

            
cv2.destroyAllWindows()


# In[ ]:




