import cv2
import os
import json

# --- SETUP DATABASE ---
def save_name(id, name):
    data = {}
    # Check if file exists, load it
    if os.path.exists('names.json'):
        with open('names.json', 'r') as f:
            try:
                data = json.load(f)
            except:
                data = {} # If file is empty/corrupt
    
    # Add new user
    data[str(id)] = name
    
    # Save back to file
    with open('names.json', 'w') as f:
        json.dump(data, f)
    print(f"\n [INFO] Saved Name: ID {id} = {name}")

# --- MAIN CODE ---
if not os.path.exists('dataset'):
    os.makedirs('dataset')

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 1. Ask for ID and Name
face_id = input('\n Enter User ID (integer, e.g., 1) ==>  ')
face_name = input(' Enter User Name (e.g., John)     ==>  ')

# 2. Save the name to our database
save_name(face_id, face_name)

print("\n [INFO] Initializing face capture. Look at the camera...")
count = 0

while(True):
    ret, img = cap.read()
    if not ret: break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        
        # Save image
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27: break
    elif count >= 30: break

print("\n [INFO] Success! Photos saved.")
cap.release()
cv2.destroyAllWindows()