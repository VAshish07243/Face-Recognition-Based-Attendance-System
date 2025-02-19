
import cv2
import face_recognition
import os
import openpyxl
from openpyxl import Workbook
from datetime import datetime

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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found and the student hasn't been recorded yet, add to the attendance sheet
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            if name not in recorded_students:
                # Add student name to the attendance sheet
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ws.append([timestamp, name])
                recorded_students.add(name)

        # Draw a rectangle and label on the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition Attendance System', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the workbook with attendance to a file
attendance_file = "attendance.xlsx"
wb.save(attendance_file)

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
