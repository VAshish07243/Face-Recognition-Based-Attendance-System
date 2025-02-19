import cv2
import face_recognition
import os

# Function to load images from a directory and create face encodings
def load_images_from_folder(folder):
    face_encodings = []
    face_names = []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img)

        if len(encoding) > 0:
            face_encodings.append(encoding[0])
            face_names.append(os.path.splitext(filename)[0])  # Use the filename as the student name

    return face_encodings, face_names

# Load student images and their corresponding name
dataset_folder ="C:\Face-Recognosition\students_dataset" # Provide the path to the folder containing student images
known_face_encodings, known_face_names = load_images_from_folder(dataset_folder)
print(known_face_names)