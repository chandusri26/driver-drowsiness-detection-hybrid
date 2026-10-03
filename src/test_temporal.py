import cv2
import dlib

from feature_extraction import (
    calculate_both_eyes_ear,
    calculate_mouth_mar
)

from drowsiness_logic import DrowsinessDetector

from alarm import Alarm


# ==================================================
# CONFIGURATION
# ==================================================

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

ALARM_FILE = "alarm.wav"

EAR_THRESHOLD = 0.25

MAR_THRESHOLD = 0.60

EYE_CLOSED_TIME = 5.0

EYE_OPEN_TIME = 3.0

YAWN_TIME = 1.0


# ==================================================
# INITIALIZE DLIB
# ==================================================

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    PREDICTOR_PATH
)


# ==================================================
# INITIALIZE DROWSINESS DETECTOR
# ==================================================

drowsiness_detector = DrowsinessDetector(

    ear_threshold=EAR_THRESHOLD,

    mar_threshold=MAR_THRESHOLD,

    eye_closed_duration=EYE_CLOSED_TIME,

    eye_open_duration=EYE_OPEN_TIME,

    yawn_duration=YAWN_TIME
)


# ==================================================
# INITIALIZE ALARM
# ==================================================

alarm = Alarm(ALARM_FILE)


# ==================================================
# START WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print("----------------------------------------")
print("Driver Drowsiness Detection Started")
print("----------------------------------------")
print("Eyes closed for 5 seconds → ALARM")
print("Eyes open for 3 seconds → ALARM STOPS")
print("Yawn → Screen alert only")
print("Press Q to quit")
print("----------------------------------------")


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read webcam frame.")

        break


    # ----------------------------------------------
    # Convert to grayscale
    # ----------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ----------------------------------------------
    # Detect faces
    # ----------------------------------------------

    faces = detector(gray)


    # ==================================================
    # NO FACE
    # ==================================================

    if len(faces) == 0:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        # If alarm is already active,
        # DON'T stop it merely because the
        # face temporarily disappeared.

        if drowsiness_detector.drowsiness_alert:

            alarm.start()


    # ==================================================
    # FACE DETECTED
    # ==================================================

    for face in faces:

        # ------------------------------------------
        # Facial landmarks
        # ------------------------------------------

        landmarks = predictor(
            gray,
            face
        )


        # ------------------------------------------
        # EAR
        # ------------------------------------------

        left_ear, right_ear, ear = calculate_both_eyes_ear(
            landmarks
        )


        # ------------------------------------------
        # MAR
        # ------------------------------------------

        mar = calculate_mouth_mar(
            landmarks
        )


        # ------------------------------------------
        # Update detector
        # ------------------------------------------

        result = drowsiness_detector.update(
            ear,
            mar
        )


        drowsy = result["drowsy"]

        yawning = result["yawning"]

        closed_duration = result["closed_duration"]

        open_duration = result["open_duration"]

        yawn_duration = result["yawn_duration"]


        # ------------------------------------------
        # Face rectangle
        # ------------------------------------------

        cv2.rectangle(
            frame,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (0, 255, 0),
            2
        )


        # ==================================================
        # EAR
        # ==================================================

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ==================================================
        # MAR
        # ==================================================

        cv2.putText(
            frame,
            f"MAR: {mar:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # ==================================================
        # EYE STATUS
        # ==================================================

        if ear < EAR_THRESHOLD:

            eye_status = "CLOSED"

        else:

            eye_status = "OPEN"


        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ==================================================
        # MOUTH STATUS
        # ==================================================

        if yawning:

            mouth_status = "YAWNING"

        else:

            mouth_status = "NORMAL"


        cv2.putText(
            frame,
            f"Mouth: {mouth_status}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ==================================================
        # CLOSED EYE TIMER
        # ==================================================

        if eye_status == "CLOSED":

            cv2.putText(
                frame,
                f"Eyes closed: {closed_duration:.1f}s / 5.0s",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )


        # ==================================================
        # OPEN EYE RECOVERY TIMER
        # ==================================================

        if drowsy and eye_status == "OPEN":

            cv2.putText(
                frame,
                f"Recovery: {open_duration:.1f}s / 3.0s",
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


        # ==================================================
        # YAWN ALERT
        # ==================================================

        if yawning and not drowsy:

            cv2.putText(
                frame,
                "YAWNING DETECTED",
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                3
            )

            cv2.putText(
                frame,
                "Please stay alert",
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )


        # ==================================================
        # DROWSINESS ALERT
        # ==================================================

        if drowsy:

            # Start alarm
            alarm.start()


            # Large red warning
            cv2.putText(
                frame,
                "!!! DROWSINESS ALERT !!!",
                (20, 365),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )


            cv2.putText(
                frame,
                "WAKE UP! PLEASE OPEN YOUR EYES",
                (20, 405),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )


        else:

            # Stop alarm only when the
            # detector has completed its
            # 3-second eye-open recovery.
            alarm.stop()


            # ------------------------------------------
            # Normal status
            # ------------------------------------------

            if not yawning:

                cv2.putText(
                    frame,
                    "DRIVER ALERT",
                    (20, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )


    # ==================================================
    # DISPLAY
    # ==================================================

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )


    # ==================================================
    # QUIT
    # ==================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==================================================
# CLEANUP
# ==================================================

alarm.stop()

cap.release()

cv2.destroyAllWindows()

print("Driver Drowsiness Detection stopped.")