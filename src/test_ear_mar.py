import cv2
import dlib

from feature_extraction import (
    calculate_both_eyes_ear,
    calculate_mouth_mar
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Initial thresholds.
# We will NOT treat these as final values yet.
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.60


# --------------------------------------------------
# Initialize dlib
# --------------------------------------------------

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    PREDICTOR_PATH
)


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


    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Detect faces
    faces = detector(gray)


    for face in faces:

        # Get 68 facial landmarks
        landmarks = predictor(
            gray,
            face
        )


        # ------------------------------------------
        # Calculate EAR
        # ------------------------------------------

        left_ear, right_ear, ear = calculate_both_eyes_ear(
            landmarks
        )


        # ------------------------------------------
        # Calculate MAR
        # ------------------------------------------

        mar = calculate_mouth_mar(
            landmarks
        )


        # ------------------------------------------
        # Determine eye state
        # ------------------------------------------

        if ear < EAR_THRESHOLD:

            eye_status = "CLOSED"

        else:

            eye_status = "OPEN"


        # ------------------------------------------
        # Determine mouth state
        # ------------------------------------------

        if mar > MAR_THRESHOLD:

            mouth_status = "YAWNING"

        else:

            mouth_status = "NORMAL"


        # ------------------------------------------
        # Draw face rectangle
        # ------------------------------------------

        cv2.rectangle(
            frame,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (0, 255, 0),
            2
        )


        # ------------------------------------------
        # Display EAR
        # ------------------------------------------

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ------------------------------------------
        # Display MAR
        # ------------------------------------------

        cv2.putText(
            frame,
            f"MAR: {mar:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # ------------------------------------------
        # Display eye status
        # ------------------------------------------

        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ------------------------------------------
        # Display mouth status
        # ------------------------------------------

        cv2.putText(
            frame,
            f"Mouth: {mouth_status}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ------------------------------------------
        # Display individual eye EAR values
        # ------------------------------------------

        cv2.putText(
            frame,
            f"Left EAR: {left_ear:.2f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )

        cv2.putText(
            frame,
            f"Right EAR: {right_ear:.2f}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )


    # ----------------------------------------------
    # Show frame
    # ----------------------------------------------

    cv2.imshow(
        "Driver Drowsiness - EAR & MAR",
        frame
    )


    # ----------------------------------------------
    # Quit
    # ----------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Program stopped.")