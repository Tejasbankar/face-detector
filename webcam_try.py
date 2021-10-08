import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

#load the model
model=load_model('Mask_detector_model.h5')

#capturing video from webcam
video_capture=cv2.VideoCapture(0)

while True:
    _,frame=video_capture.read()
    (h,w) = frame.shape[:2]

    #Loading the face detector model
    prototxtpath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtpath, weightsPath)

    #detecting faces in the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), swapRB=True, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    #Loop through all the faces detected
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        #Checking model confidence for detected faces
        if confidence > 0.16:

            # Getting top left  and bottom right co-ordinates of detected face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            diffX = endX - startX
            diffY = endY - startY

            # Checking if the face detected is big enough to predict
            if diffX > 0 and diffY > 0:

                # Preprocessing
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                #predicting mask
                prediction = model.predict(face)

                #showing output in the video
                if (prediction[0][1] > 0.5):
                    pred = 'Mask'
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0),2)
                    cv2.putText(frame, pred, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    pred = 'No Mask'
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255),2)
                    cv2.putText(frame, pred, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #displaying video and exit on press q
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

#destroying all windows after exit
video_capture.release()
cv2.destroyAllWindows()
