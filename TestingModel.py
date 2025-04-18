'''
from keras.models import model_from_json
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


emotion_dict={0:"angry",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad",6:"surprise"}

json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()

emotion_model=model_from_json(loaded_model_json)

emotion_model.load_weights("model.weights.h5")
print("loaded  model disk")



cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


#cap=cv2.VideoCapture('test.mp4')

while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(48,48))
    if not ret:
        break
    face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces on camera
    num_faces=face_detector.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors=5)

    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi_gray_frame=gray_frame[y:y+h,x:x+w]
        cropped_img=np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,(48,48)),-1),0)

        emotion_prediction=emotion_model.predict(cropped_img)
        maxindex=int(np.argmax(emotion_prediction))
        cv2.putText(frame,emotion_dict[maxindex],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
'''

# Suppress TensorFlow logs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import model_from_json
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte  # Import for converting to uint8



def compute_lbp(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)  
        image = img_as_ubyte(image)  
    else:
        image = img_as_ubyte(image)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp




# Emotion dictionary
emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

# Load model architecture and weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model.weights.h5")
print("Loaded model from disk")

# Start video capture
#cap = cv2.VideoCapture(0)
cap=cv2.VideoCapture('test.mp4')

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x + w, y + h + 10), (0, 255, 0), 4)
        
        # Extract the region of interest (ROI) for emotion prediction
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict emotion
        # Compute the LBP features for the cropped image
        lbp_features = compute_lbp(cropped_img.squeeze())    # Assuming you have a function to compute LBP
        lbp_features = lbp_features[:26, :]                  # Adjust the shape if necessary
        lbp_features = np.expand_dims(lbp_features, axis=0)  # Add batch dimension
        emotion_prediction = emotion_model.predict([cropped_img, lbp_features])
        maxindex = int(np.argmax(emotion_prediction))

        # Display the emotion on the frame
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disable oneDNN

import cv2
import numpy as np
import keyboard  # For key press detection, install with `pip install keyboard`
from keras.models import model_from_json
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte

# Function to compute Local Binary Pattern (LBP)
def compute_lbp(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)
        image = img_as_ubyte(image)
    else:
        image = img_as_ubyte(image)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp

# Emotion labels dictionary
emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

# Load model architecture and weights
try:
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model.weights.h5")
    print("Loaded model from disk successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('test.mp4')

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Compute LBP features
        lbp_features = compute_lbp(cropped_img.squeeze())
        lbp_features = np.expand_dims(lbp_features[:26, :], axis=0)  # Adjust shape if required

        try:
            # Predict emotion
            emotion_prediction = emotion_model.predict([cropped_img, lbp_features])
            maxindex = int(np.argmax(emotion_prediction))
            # Display the emotion on the frame
            cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Prediction error: {e}")
            break

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard  # For key press detection, install with `pip install keyboard`
from keras.models import model_from_json
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte

# Suppress TensorFlow INFO logs and disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to compute Local Binary Pattern (LBP)
def compute_lbp(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)
        image = img_as_ubyte(image)
    else:
        image = img_as_ubyte(image)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp

# Emotion labels dictionary
emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

# Load model architecture and weights
try:
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model.weights.h5")
    print("Loaded model from disk successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# Initialize video capture
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('test.mp4')

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to display the frame using matplotlib
def show_frame_matplotlib(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)
    plt.clf()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Compute LBP features
        lbp_features = compute_lbp(cropped_img.squeeze())
        lbp_features = np.expand_dims(lbp_features[:26, :], axis=0)  # Adjust shape if required

        try:
            # Predict emotion
            emotion_prediction = emotion_model.predict([cropped_img, lbp_features])
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Prediction error: {e}")
            break

    # Display the resulting frame using matplotlib
    show_frame_matplotlib(frame)

    # Break the loop if 'q' is pressed
    if keyboard.is_pressed('q'):
        print("Quitting...")
        break

# Release the capture
cap.release()
plt.close()
'''