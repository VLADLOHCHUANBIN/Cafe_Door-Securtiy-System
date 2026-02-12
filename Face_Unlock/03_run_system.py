import cv2
import numpy as np
import time
import json
import os

# --- LOAD NAMES DATABASE ---
names = {}
if os.path.exists('names.json'):
    with open('names.json', 'r') as f:
        names = json.load(f)
    print(f" [INFO] Loaded {len(names)} names from database.")
else:
    print(" [WARNING] No names.json found! Showing IDs only.")

# --- SETUP ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Variables
door_unlocked = False
unlock_timer = 0
REQ_CONFIDENCE = 60 

print("\n [INFO] System Active...")

while True:
    ret, img = cap.read()
    if not ret: break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(0.1*cap.get(3)), int(0.1*cap.get(4))),
    )

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if (confidence < REQ_CONFIDENCE):
            # LOOK UP NAME AUTOMATICALLY
            # Convert ID to string because JSON keys are strings
            id_name = names.get(str(id), f"ID:{id}") 
            
            confidence_text = f"  {round(100 - confidence)}%"
            
            if not door_unlocked:
                door_unlocked = True
                unlock_timer = time.time()
                print(f">>> UNLOCKING DOOR FOR: {id_name}")
            
            color = (0, 255, 0)
        else:
            id_name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"
            color = (0, 0, 255)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, str(id_name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    if door_unlocked:
        time_left = 3.0 - (time.time() - unlock_timer)
        if time_left > 0:
            cv2.putText(img, f"ACCESS GRANTED ({time_left:.1f}s)", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            door_unlocked = False
            print(">>> LOCKING DOOR")

    cv2.imshow('Face Security', img)
    if cv2.waitKey(10) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()