import cv2
import numpy as np
import time
import json
import os
import RPi.GPIO as GPIO  

# --- 1. HARDWARE SETUP ---
RELAY_PIN = 17  
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW) # Locked by default

# --- 2. SETUP FACE SYSTEM ---
print(" [INFO] Loading Face System...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    # Loads the 'Brain' from your trainer folder
    recognizer.read('trainer/trainer.yml') 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    print(f" [ERROR] Could not load face files: {e}")
    exit()

# Load Name Database
names = {}
if os.path.exists('names.json'):
    with open('names.json', 'r') as f:
        names = json.load(f)

# --- 3. SETUP CARD SYSTEM (ORB) ---
print(" [INFO] Loading Card System...")
template_path = 'template.jpg'
if os.path.exists(template_path):
    template = cv2.imread(template_path, 0)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(template, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    card_system_active = True
else:
    print(" [WARNING] 'template.jpg' missing. Card mode DISABLED.")
    card_system_active = False

# --- 4. SETTINGS & THRESHOLDS ---
cap = cv2.VideoCapture(0)
cap.set(3, 320) # Lower resolution for Pi 3 speed
cap.set(4, 240)

# STRICTER THRESHOLDS TO PREVENT RELAY CLICKING
REQ_FACE_CONFIDENCE = 45  # Lower is stricter for LBPH
REQ_CARD_MATCHES = 20     # Higher is stricter for ORB

door_unlocked = False
unlock_timer = 0

print(f"\n [SYSTEM ONLINE] Security Active. Press 'q' to exit.")

try:
    while True:
        ret, img = cap.read()
        if not ret: break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        current_time = time.time()

        # ==========================
        # FACE RECOGNITION LOGIC
        # ==========================
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # If match is strong enough
            if confidence < REQ_FACE_CONFIDENCE:
                name = names.get(str(id), f"ID:{id}")
                if not door_unlocked:
                    print(f">>> FACE ACCEPTED: Welcome {name}")
                    GPIO.output(RELAY_PIN, GPIO.HIGH) # UNLOCK
                    door_unlocked = True
                    unlock_timer = current_time
                
                # Draw on the pop-out window
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, name, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
                cv2.putText(img, "Unknown", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # ==========================
        # CARD RECOGNITION LOGIC
        # ==========================
        if card_system_active and not door_unlocked:
            kp2, des2 = orb.detectAndCompute(gray, None)
            if des2 is not None:
                matches = bf.match(des1, des2)
                good_matches = [m for m in matches if m.distance < 35]
                
                if len(good_matches) > REQ_CARD_MATCHES:
                    print(f">>> CARD ACCEPTED")
                    GPIO.output(RELAY_PIN, GPIO.HIGH) # UNLOCK
                    door_unlocked = True
                    unlock_timer = current_time

        # ==========================
        # DOOR LOCK TIMER
        # ==========================
        if door_unlocked:
            if current_time - unlock_timer > 3.0: 
                print(">>> Locking Door")
                GPIO.output(RELAY_PIN, GPIO.LOW) # LOCK
                door_unlocked = False

        # ==========================
        # THE POP-OUT WINDOW
        # ==========================
        cv2.imshow('Cafe Access Control', img)
        
        # Mandatory for window to update and detect 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    print("\n [INFO] Cleaning up...")
    GPIO.output(RELAY_PIN, GPIO.LOW)
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
