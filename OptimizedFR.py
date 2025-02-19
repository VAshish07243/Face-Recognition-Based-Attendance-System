import cv2
import face_recognition
import os

# Function to load images from a directory and create face encoding
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
dataset_folder = "C:\\Face-Recognition\\students_dataset"  # Provide the path to the folder containing student images
known_face_encodings, known_face_names = load_images_from_folder(dataset_folder)

# Open a video stream (you may need to adjust the argument to your camera)
cap = cv2.VideoCapture(0)

# Function to resize frame while maintaining aspect ratio
def resize_frame(frame, width=None, height=None):
    r = width / frame.shape[1]
    dim = (width, int(frame.shape[0] * r))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Variable to control face encoding frequency
count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to improve processing speed
    resized_frame = resize_frame(frame, width=640)

    # Use OpenCV for resizing and face detection with the cnn model
    small_frame = cv2.resize(resized_frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")

    # Process face encodings at a reduced frequency
    if count % 5 == 0:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle and label on the face in the original frame
        cv2.rectangle(frame, (left * int(frame.shape[1] / resized_frame.shape[1]), 
                              top * int(frame.shape[0] / resized_frame.shape[0])),
                      (right * int(frame.shape[1] / resized_frame.shape[1]), 
                       bottom * int(frame.shape[0] / resized_frame.shape[0])), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left * int(frame.shape[1] / resized_frame.shape[1]) + 6, 
                                  bottom * int(frame.shape[0] / resized_frame.shape[0]) - 6), 
                    font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition Attendance System', frame)

    # Increment the counter
    count += 1

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

    
