import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

# Create the Recognizer and Detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        # Convert to grayscale
        try:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            
            # Get the ID from the filename (User.1.1.jpg)
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        except:
            continue # Skip files that aren't images

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model
if not os.path.exists('trainer'):
    os.makedirs('trainer')

recognizer.write('trainer/trainer.yml') 

print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")