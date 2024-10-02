import cv2
import dlib
import numpy as np
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def get_face_embedding(image_path):
    
    image = cv2.imread(image_path)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    dets = detector(rgb_image, 1)
    
    if len(dets) == 0:
        raise ValueError("No face detected in the image.")
    
    # Assume only one face per image for simplicity
    shape = sp(rgb_image, dets[0])
    
    # Get the face embedding (128-dimensional vector)
    face_embedding = np.array(facerec.compute_face_descriptor(rgb_image, shape))
    
    return face_embedding

def compare_faces(embedding1, embedding2):
    # Compute Euclidean distance between the two embeddings
    distance_between_faces = distance.euclidean(embedding1, embedding2)
    
    return distance_between_faces

try:
    # Get face embeddings for both images
    embedding1 = get_face_embedding('detected_faces/face_1.jpg')
    embedding2 = get_face_embedding('detected_faces/face_2.jpg')
    
    # Compare the embeddings
    distance_between_faces = compare_faces(embedding1, embedding2)
    
    # A threshold value for face similarity (you can adjust this threshold)
    threshold = 0.6
    
    if distance_between_faces < threshold:
        print(f"The faces match with a distance of {distance_between_faces:.4f}")
    else:
        print(f"The faces do not match with a distance of {distance_between_faces:.4f}")

except ValueError as e:
    print(e)
