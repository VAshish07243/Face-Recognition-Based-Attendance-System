import cv2
import face_recognition
import os
import openpyxl
from openpyxl import Workbook
from datetime import datetime
import threading

# Function to load images from a directory and create face encoding
def load_images_from_folder(folder):
    face_encodings = []
    face_names = []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img, model='small')

        if len(encoding) > 0:
            face_encodings.append(encoding[0])
            face_names.append(os.path.splitext(filename)[0])  # Use the filename as the student name

    return face_encodings, face_names

# Load student images and their corresponding names
dataset_folder = r"C:\Face-Recognition\students_dataset"  # Provide the path to the folder containing student images
known_face_encodings, known_face_names = load_images_from_folder(dataset_folder)

# Open a video stream (you may need to adjust the argument to your camera)
cap = cv2.VideoCapture(0)

# Create a workbook and add a sheet for attendance
wb = Workbook()
ws = wb.active
ws.append(["Timestamp", "Student Name"])

# Set to keep track of recorded students
recorded_students = set()

# Lock for thread safety
lock = threading.Lock()

# Tolerance for face matching precision
face_match_tolerance = 0.5

# Function to perform face recognition in a separate thread
def recognize_faces(frame):
    global recorded_students
    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(frame_small)
    face_encodings = face_recognition.face_encodings(frame_small, face_locations, model='small')

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        min_distance_index = distances.argmin()

        if distances[min_distance_index] < face_match_tolerance:
            name = known_face_names[min_distance_index]

            with lock:
                if name not in recorded_students:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ws.append([timestamp, name])
                    recorded_students.add(name)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# Function to read frames from the camera
def read_frames():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        recognize_faces(frame)
        cv2.imshow('Face Recognition Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start a separate thread for face recognition
face_recognition_thread = threading.Thread(target=read_frames)
face_recognition_thread.start()

# Wait for the face recognition thread to finish
face_recognition_thread.join()

# Save the workbook with attendance to a file
attendance_file = "attendance.xlsx"
wb.save(attendance_file)

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()



