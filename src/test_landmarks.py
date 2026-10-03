import cv2
import dlib


# --------------------------------------------------
# Paths
# --------------------------------------------------
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


# --------------------------------------------------
# Initialize dlib
# --------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# --------------------------------------------------
# Start webcam
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press 'q' to quit.")


# --------------------------------------------------
# Main loop
# --------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Convert BGR → grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    # Process every detected face
    for face in faces:

        # Get 68 facial landmarks
        landmarks = predictor(gray, face)

        # Draw face rectangle
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw all 68 landmarks
        for i in range(68):

            x = landmarks.part(i).x
            y = landmarks.part(i).y

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 0, 255),
                -1
            )

            # Display landmark number
            cv2.putText(
                frame,
                str(i),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1
            )

        # Display number of detected faces
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show webcam
    cv2.imshow(
        "Driver Drowsiness - Facial Landmarks",
        frame
    )

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("Program stopped.")