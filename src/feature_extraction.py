import numpy as np


# --------------------------------------------------
# Calculate Euclidean distance
# --------------------------------------------------
def euclidean_distance(point1, point2):
    """
    Calculate distance between two (x, y) points.
    """
    return np.linalg.norm(
        np.array(point1) - np.array(point2)
    )


# --------------------------------------------------
# Extract landmark coordinates
# --------------------------------------------------
def get_landmark_points(landmarks, start, end):
    """
    Extract landmark coordinates from start to end.

    Example:
        get_landmark_points(landmarks, 36, 41)
        returns the six points of an eye.
    """
    points = []

    for i in range(start, end + 1):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append((x, y))

    return points


# --------------------------------------------------
# Eye Aspect Ratio (EAR)
# --------------------------------------------------
def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio.

    eye_points must contain 6 points:

        p1 ---- p2
      /            \
    p6              p3
      \            /
        p5 ---- p4

    EAR = (|p2-p6| + |p3-p5|)
          ---------------------
               2 * |p1-p4|
    """

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    # Prevent division by zero
    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# --------------------------------------------------
# Mouth Aspect Ratio (MAR)
# --------------------------------------------------
def calculate_mar(mouth_points):
    """
    Calculate Mouth Aspect Ratio.

    Uses the 20 mouth landmarks (48-67).

    We use:
        vertical distances between upper/lower mouth
        horizontal mouth width
    """

    # Outer mouth landmarks:
    #
    # 48 ---------------- 54
    #    49             53
    #      50         52
    #        51
    #
    #        57
    #      56   58
    #    55       59
    #  54?       ...
    #
    # For MAR we use selected vertical/horizontal distances.

    horizontal = euclidean_distance(
        mouth_points[0],
        mouth_points[6]
    )

    vertical_1 = euclidean_distance(
        mouth_points[2],
        mouth_points[10]
    )

    vertical_2 = euclidean_distance(
        mouth_points[4],
        mouth_points[8]
    )

    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return mar


# --------------------------------------------------
# Get both eye EAR values
# --------------------------------------------------
def calculate_both_eyes_ear(landmarks):
    """
    Calculate EAR for both eyes.

    Left eye:
        landmarks 36-41

    Right eye:
        landmarks 42-47
    """

    left_eye = get_landmark_points(
        landmarks,
        36,
        41
    )

    right_eye = get_landmark_points(
        landmarks,
        42,
        47
    )

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    average_ear = (left_ear + right_ear) / 2.0

    return left_ear, right_ear, average_ear


# --------------------------------------------------
# Calculate MAR from facial landmarks
# --------------------------------------------------
def calculate_mouth_mar(landmarks):
    """
    Extract mouth landmarks 48-67
    and calculate MAR.
    """

    mouth = get_landmark_points(
        landmarks,
        48,
        67
    )

    return calculate_mar(mouth)