import cv2
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image_path = 'two_faces_1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert the image to grayscale (Haar Cascade works better on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Create a directory to save the extracted faces
output_dir = 'detected_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the detected faces and save each as a separate image
for i, (x, y, w, h) in enumerate(faces):
    # Extract the face from the image
    face = image[y:y+h, x:x+w]
    
    # Save the face image
    face_filename = os.path.join(output_dir, f'face_{i+1}.jpg')
    cv2.imwrite(face_filename, face)

    print(f'Saved {face_filename}')

print(f'Total faces detected: {len(faces)}')


