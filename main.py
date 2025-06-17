import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces
student1_image = face_recognition.load_image_file("faces/student1.jpeg")
student1_encoding = face_recognition.face_encodings(student1_image)[0]

student2_image = face_recognition.load_image_file("faces/student2.jpeg")
student2_encoding = face_recognition.face_encodings(student2_image)[0]

known_face_encodings = [student1_encoding, student2_encoding]
known_face_names = ["Student1", "Student2"]

# List of expected students
students = known_face_names.copy()

# Create CSV with today's date
now = datetime.now()
filename = now.strftime("%Y-%m-%d") + ".csv"
f = open(filename, "w", newline="")
lnwriter = csv.writer(f)

# Write header row
lnwriter.writerow(["Name", "Time"])

process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = []
    face_encodings = []

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distance) > 0:
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name + " Present", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Log attendance
        if name in students:
            students.remove(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            lnwriter.writerow([name, current_time])
            print(f"Marked {name} present at {current_time}")

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
