import os
import time

import cv2
import dlib
import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

from feature_extraction import (
    calculate_both_eyes_ear,
    calculate_mouth_mar
)

from alarm import Alarm

from cnn_predictor import (
    EyeCNNPredictor,
    YawnCNNPredictor
)

from hybrid_fusion import HybridFusion

from hybrid_drowsiness_logic import (
    HybridDrowsinessDetector
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PREDICTOR_PATH = os.path.join(
    ROOT_DIR,
    "shape_predictor_68_face_landmarks.dat"
)

EYE_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "eye_mobilenetv2.keras"
)

YAWN_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "yawn_mobilenetv2.keras"
)

ALARM_PATH = os.path.join(
    ROOT_DIR,
    "alarm.wav"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    PREDICTOR_PATH,
    EYE_MODEL_PATH,
    YAWN_MODEL_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# INITIALIZE DLIB
# ============================================================

print("Loading dlib face detector...")

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    PREDICTOR_PATH
)


# ============================================================
# LOAD CNN MODELS
# ============================================================

print("Loading Eye CNN...")

eye_cnn = EyeCNNPredictor(
    EYE_MODEL_PATH
)

print("✓ Eye CNN loaded")


print("Loading Yawn CNN...")

yawn_cnn = YawnCNNPredictor(
    YAWN_MODEL_PATH
)

print("✓ Yawn CNN loaded")


# ============================================================
# HYBRID COMPONENTS
# ============================================================

fusion = HybridFusion(
    ear_threshold=0.25,
    mar_threshold=0.60,
    eye_fusion_threshold=0.60,
    yawn_fusion_threshold=0.60
)

logic = HybridDrowsinessDetector(
    eye_closed_duration=5.0,
    eye_open_duration=3.0,
    yawn_duration=1.0
)

alarm = Alarm(
    ALARM_PATH
)


# ============================================================
# CNN INFERENCE SETTINGS
# ============================================================

# CNN inference is performed every N frames.
#
# Frame 1 → CNN inference
# Frame 2 → reuse previous CNN result
# Frame 3 → CNN inference
# Frame 4 → reuse previous CNN result
#
# EAR and MAR are still calculated every frame.

CNN_INTERVAL = 2


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


print("\n")
print("=" * 70)
print("OPTIMIZED HYBRID DRIVER DROWSINESS DETECTION")
print("=" * 70)

print(
    f"\nCNN inference interval: "
    f"every {CNN_INTERVAL} frames"
)

print("\nPress Q to quit.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_eye_crop(
    frame,
    shape,
    start,
    end,
    margin=10
):

    points = np.array([
        (
            shape.part(i).x,
            shape.part(i).y
        )
        for i in range(start, end)
    ])

    x_min = max(
        0,
        np.min(points[:, 0]) - margin
    )

    y_min = max(
        0,
        np.min(points[:, 1]) - margin
    )

    x_max = min(
        frame.shape[1],
        np.max(points[:, 0]) + margin
    )

    y_max = min(
        frame.shape[0],
        np.max(points[:, 1]) + margin
    )

    return frame[
        y_min:y_max,
        x_min:x_max
    ]


def get_mouth_crop(
    frame,
    shape,
    margin=15
):

    points = np.array([
        (
            shape.part(i).x,
            shape.part(i).y
        )
        for i in range(48, 68)
    ])

    x_min = max(
        0,
        np.min(points[:, 0]) - margin
    )

    y_min = max(
        0,
        np.min(points[:, 1]) - margin
    )

    x_max = min(
        frame.shape[1],
        np.max(points[:, 0]) + margin
    )

    y_max = min(
        frame.shape[0],
        np.max(points[:, 1]) + margin
    )

    return frame[
        y_min:y_max,
        x_min:x_max
    ]


def draw_text(
    frame,
    text,
    position,
    scale=0.65,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# PERFORMANCE MEASUREMENT
# ============================================================

benchmark_start_time = time.perf_counter()

previous_time = time.perf_counter()

fps = 0.0

frame_count = 0

fps_values = []


# ============================================================
# CNN CACHE
# ============================================================

cached_eye_open_probability = 1.0

cached_yawn_probability = 0.0

cnn_frame_counter = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Could not read webcam frame."
        )

        break


    frame_count += 1


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.perf_counter()

    elapsed = (
        current_time -
        previous_time
    )

    if elapsed > 0:

        fps = 1.0 / elapsed

        fps_values.append(
            fps
        )

    previous_time = current_time


    # ========================================================
    # FACE DETECTION
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector(
        gray,
        0
    )


    # ========================================================
    # NO FACE
    # ========================================================

    if len(faces) == 0:

        draw_text(
            frame,
            "NO FACE DETECTED",
            (30, 40),
            0.8,
            2
        )

        draw_text(
            frame,
            f"FPS: {fps:.1f}",
            (
                30,
                frame.shape[0] - 20
            )
        )

        cv2.imshow(
            "Hybrid Driver Drowsiness Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        continue


    # ========================================================
    # SELECT LARGEST FACE
    # ========================================================

    face = max(
        faces,
        key=lambda rect:
            rect.width() *
            rect.height()
    )


    # ========================================================
    # LANDMARKS
    # ========================================================

    shape = predictor(
        gray,
        face
    )


    # ========================================================
    # FACE RECTANGLE
    # ========================================================

    cv2.rectangle(
        frame,
        (
            face.left(),
            face.top()
        ),
        (
            face.right(),
            face.bottom()
        ),
        (0, 255, 0),
        2
    )


    # ========================================================
    # GEOMETRIC FEATURES
    # ========================================================

    ear, left_ear, right_ear = (
        calculate_both_eyes_ear(
            shape
        )
    )

    mar = calculate_mouth_mar(
        shape
    )


    # ========================================================
    # CNN INFERENCE
    # ========================================================

    cnn_frame_counter += 1

    should_run_cnn = (
        cnn_frame_counter == 1
        or
        cnn_frame_counter >= CNN_INTERVAL
    )


    if should_run_cnn:

        cnn_frame_counter = 0


        # ----------------------------------------------------
        # EXTRACT EYE CROPS
        # ----------------------------------------------------

        left_eye_crop = get_eye_crop(
            frame,
            shape,
            36,
            42
        )

        right_eye_crop = get_eye_crop(
            frame,
            shape,
            42,
            48
        )


        # ----------------------------------------------------
        # BATCH BOTH EYES INTO ONE CNN CALL
        # ----------------------------------------------------

        eye_predictions = (
            eye_cnn.predict_batch(
                [
                    left_eye_crop,
                    right_eye_crop
                ]
            )
        )


        if eye_predictions is not None:

            left_prediction = (
                eye_predictions[0]
            )

            right_prediction = (
                eye_predictions[1]
            )


            cached_eye_open_probability = (
                left_prediction[
                    "open_probability"
                ]
                +
                right_prediction[
                    "open_probability"
                ]
            ) / 2.0


        # ----------------------------------------------------
        # MOUTH CROP
        # ----------------------------------------------------

        mouth_crop = get_mouth_crop(
            frame,
            shape
        )


        # ----------------------------------------------------
        # YAWN CNN
        # ----------------------------------------------------

        yawn_prediction = (
            yawn_cnn.predict(
                mouth_crop
            )
        )


        if yawn_prediction is not None:

            cached_yawn_probability = (
                yawn_prediction[
                    "yawn_probability"
                ]
            )


    # ========================================================
    # USE CACHED CNN RESULTS
    # ========================================================

    eye_open_probability = (
        cached_eye_open_probability
    )

    eye_closed_probability = (
        1.0 -
        eye_open_probability
    )

    yawn_probability = (
        cached_yawn_probability
    )


    # ========================================================
    # HYBRID FUSION
    # ========================================================

    eye_result = (
        fusion.calculate_eye_fusion(
            ear,
            eye_closed_probability
        )
    )


    yawn_result = (
        fusion.calculate_yawn_fusion(
            mar,
            yawn_probability
        )
    )


    eyes_closed = (
        eye_result[
            "eyes_closed"
        ]
    )

    yawning = (
        yawn_result[
            "yawning"
        ]
    )


    # ========================================================
    # TEMPORAL LOGIC
    # ========================================================

    status = logic.update(
        eyes_closed,
        yawning
    )


    # ========================================================
    # ALARM
    # ========================================================

    if status[
        "drowsiness_alert"
    ]:

        alarm.start()

    else:

        alarm.stop()


    # ========================================================
    # DISPLAY VALUES
    # ========================================================

    draw_text(
        frame,
        f"EAR: {ear:.2f}",
        (30, 40)
    )

    draw_text(
        frame,
        f"MAR: {mar:.2f}",
        (30, 70)
    )

    draw_text(
        frame,
        (
            f"Eye CNN Open: "
            f"{eye_open_probability:.2f}"
        ),
        (30, 100)
    )

    draw_text(
        frame,
        (
            f"Yawn CNN: "
            f"{yawn_probability:.2f}"
        ),
        (30, 130)
    )

    draw_text(
        frame,
        (
            f"Eye Fusion: "
            f"{eye_result['eye_fusion_score']:.2f}"
        ),
        (30, 160)
    )

    draw_text(
        frame,
        (
            f"Yawn Fusion: "
            f"{yawn_result['yawn_fusion_score']:.2f}"
        ),
        (30, 190)
    )


    # ========================================================
    # EYE STATUS
    # ========================================================

    if eyes_closed:

        draw_text(
            frame,
            "EYES: CLOSED",
            (30, 230)
        )

        draw_text(
            frame,
            (
                f"Closed: "
                f"{status['closed_duration']:.1f}s"
            ),
            (30, 260)
        )

    else:

        draw_text(
            frame,
            "EYES: OPEN",
            (30, 230)
        )


    # ========================================================
    # YAWN STATUS
    # ========================================================

    if status["yawning"]:

        draw_text(
            frame,
            "YAWNING DETECTED",
            (30, 300),
            0.8,
            2
        )

        draw_text(
            frame,
            "PLEASE STAY ALERT",
            (30, 335),
            0.8,
            2
        )


    # ========================================================
    # DROWSINESS ALERT
    # ========================================================

    if status[
        "drowsiness_alert"
    ]:

        draw_text(
            frame,
            "!!! DROWSINESS ALERT !!!",
            (30, 390),
            0.9,
            3
        )

        draw_text(
            frame,
            "WAKE UP! PLEASE OPEN YOUR EYES",
            (30, 430),
            0.75,
            2
        )

    else:

        draw_text(
            frame,
            "DRIVER ALERT",
            (30, 390),
            0.8,
            2
        )


    # ========================================================
    # FPS DISPLAY
    # ========================================================

    draw_text(
        frame,
        f"FPS: {fps:.1f}",
        (
            30,
            frame.shape[0] - 20
        )
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Hybrid Driver Drowsiness Detection",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

alarm.stop()

cap.release()

cv2.destroyAllWindows()


# ============================================================
# PERFORMANCE REPORT
# ============================================================

benchmark_end_time = time.perf_counter()

total_runtime = (
    benchmark_end_time -
    benchmark_start_time
)

average_fps = (
    frame_count / total_runtime
    if total_runtime > 0
    else 0.0
)


if fps_values:

    minimum_fps = min(
        fps_values
    )

    maximum_fps = max(
        fps_values
    )

else:

    minimum_fps = 0.0

    maximum_fps = 0.0


print("\n")
print("=" * 70)
print("OPTIMIZED REAL-TIME PERFORMANCE REPORT")
print("=" * 70)

print(
    f"Total runtime : "
    f"{total_runtime:.2f} seconds"
)

print(
    f"Frames        : "
    f"{frame_count}"
)

print(
    f"Average FPS   : "
    f"{average_fps:.2f}"
)

print(
    f"Minimum FPS   : "
    f"{minimum_fps:.2f}"
)

print(
    f"Maximum FPS   : "
    f"{maximum_fps:.2f}"
)

print(
    f"CNN interval  : "
    f"Every {CNN_INTERVAL} frames"
)

print("=" * 70)

print(
    "\nOptimized hybrid detection stopped."
)

