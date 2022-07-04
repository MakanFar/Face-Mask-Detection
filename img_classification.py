import keras
from PIL import Image, ImageOps
import numpy as np
import cv2
from scipy.spatial import distance
import matplotlib.pyplot as plt

mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)}
MIN_DISTANCE = 130

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)
    face_model = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4)

    label = [0 for i in range(len(faces))]
    for i in range(len(faces)-1):
        for j in range(i+1, len(faces)):
            dist = distance.euclidean(faces[i][:2],faces[j][:2])
            if dist<MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        crop = img[y:y+h,x:x+w]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = model.predict(crop)
        cv2.putText(img,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
        cv2.rectangle(img,(x,y),(x+w,y+h),dist_label[label[i]],1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    fig1 = plt.gcf()
    plt.imshow(img)
    fig1.savefig('output.png', dpi=100)
    