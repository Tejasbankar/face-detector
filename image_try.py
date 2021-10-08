import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


# path of the image
filepath = "sample.jfif"

# Load the saved model
model = load_model('Mask_detector_model.h5')

#Reading the image and it's height and width
image = cv2.imread(filepath)
(h, w) = image.shape[:2]

#Loading the face detector model
prototxtpath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtpath,weightsPath)
print("[Info] Loading mask detector model")

#detecting faces in the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), swapRB=True, crop=False)
faceNet.setInput(blob)
detections = faceNet.forward()

#Loop through all the faces detected
for i in range(0,detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    #Checking the model cofindence for detected face
    if confidence > 0.18:
        #Getting top left  and bottom right co-ordinates of detected face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        diffX = endX-startX
        diffY = endY-startY

        #Checking if the face detected is big enough to predict
        if diffX>10 and diffY>10:

            #Preprocessing
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            #Predicting mask
            prediction = model.predict(face)

            if (prediction[0][1] > 0.5):
                pred = 'Mask'
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, pred, (startX, startY), cv2.QT_FONT_NORMAL, 1, (0, 255, 0), 2)
            else:
                pred = 'No Mask'
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image, pred, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

while True:
    cv2.resize(image, (1280,960))
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Face Detection', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Face Detection', image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()