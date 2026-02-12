import cv2
import numpy as np
import time
import json
import os

# --- 1. SETUP FACE SYSTEM ---
print(" [INFO] Loading Face System...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read('trainer/trainer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    print(" [ERROR] Could not load 'trainer.yml'.")
    exit()

names = {}
if os.path.exists('names.json'):
    with open('names.json', 'r') as f:
        names = json.load(f)

# --- 2. SETUP CARD SYSTEM ---
print(" [INFO] Loading Card System...")
template_path = 'template.jpg' 
if os.path.exists(template_path):
    template = cv2.imread(template_path, 0)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(template, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    card_system_active = True
    hT, wT = template.shape
else:
    card_system_active = False

cap = cv2.VideoCapture(0)

# --- CRITICAL FIX HERE ---
# Lower this number to make it STRICTER
# 60 = Loose (Mistakes strangers for you)
# 45 = Strict (Might reject you if lighting is bad, but safer)
REQ_FACE_CONFIDENCE = 45  
REQ_CARD_MATCHES = 15     

door_unlocked = False
unlock_timer = 0

print(f"\n [SYSTEM ONLINE] Strict Mode Active (<{REQ_FACE_CONFIDENCE})")

while True:
    ret, img = cap.read()
    if not ret: break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    current_time = time.time()
    
    # 1. FACE CHECK
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Calculate how "sure" the robot is (0% to 100%)
        # Note: 'confidence' variable is actually 'difference' (lower is better)
        match_percentage = round(100 - confidence)

        if confidence < REQ_FACE_CONFIDENCE:
            name = names.get(str(id), f"ID:{id}")
            
            if not door_unlocked:
                door_unlocked = True
                unlock_timer = current_time
                print(f">>> FACE UNLOCK: {name} (Match: {match_percentage}%)")
            
            color = (0, 255, 0) # Green
        else:
            name = "Unknown"
            color = (0, 0, 255) # Red
            
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{name} ({match_percentage}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 2. CARD CHECK
    if card_system_active and not door_unlocked:
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des2 is not None:
            matches = bf.match(des1, des2)
            good_matches = [m for m in matches if m.distance < 35]
            score = len(good_matches)
            
            if score > REQ_CARD_MATCHES:
                door_unlocked = True
                unlock_timer = current_time
                print(f">>> CARD UNLOCK: Score {score}")
                
                # Draw Card Box
                try:
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        pts = np.float32([ [0,0],[0,hT-1],[wT-1,hT-1],[wT-1,0] ]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts, M)
                        cv2.polylines(img, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                except: pass

    # DOOR TIMER
    if door_unlocked:
        time_left = 3.0 - (current_time - unlock_timer)
        if time_left > 0:
            cv2.putText(img, f"UNLOCKED ({time_left:.1f}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            door_unlocked = False
    else:
        cv2.putText(img, "LOCKED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Hybrid Security', img)
    if cv2.waitKey(10) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()