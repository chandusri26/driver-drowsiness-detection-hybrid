import time


class HybridDrowsinessDetector:

    def __init__(
        self,
        eye_closed_duration=5.0,
        eye_open_duration=3.0,
        yawn_duration=1.0
    ):

        self.eye_closed_duration = (
            eye_closed_duration
        )

        self.eye_open_duration = (
            eye_open_duration
        )

        self.yawn_duration = (
            yawn_duration
        )

        self.eye_closed_start = None
        self.eye_open_start = None
        self.yawn_start = None

        self.drowsiness_alert = False
        self.yawning = False

    def update(
        self,
        eyes_closed,
        yawning
    ):

        current_time = time.monotonic()

        # ==================================================
        # EYE CLOSURE / DROWSINESS
        # ==================================================

        if eyes_closed:

            self.eye_open_start = None

            if self.eye_closed_start is None:

                self.eye_closed_start = (
                    current_time
                )

            closed_duration = (
                current_time
                -
                self.eye_closed_start
            )

            if (
                closed_duration
                >= self.eye_closed_duration
            ):

                self.drowsiness_alert = True

        else:

            self.eye_closed_start = None

            closed_duration = 0.0

            # Recover only after continuous
            # eye opening for 3 seconds

            if self.drowsiness_alert:

                if self.eye_open_start is None:

                    self.eye_open_start = (
                        current_time
                    )

                open_duration = (
                    current_time
                    -
                    self.eye_open_start
                )

                if (
                    open_duration
                    >= self.eye_open_duration
                ):

                    self.drowsiness_alert = False

                    self.eye_open_start = None

            else:

                self.eye_open_start = None

                open_duration = 0.0

        # ==================================================
        # YAWN DETECTION
        # ==================================================

        if yawning:

            if self.yawn_start is None:

                self.yawn_start = (
                    current_time
                )

            yawn_duration = (
                current_time
                -
                self.yawn_start
            )

            self.yawning = (
                yawn_duration
                >= self.yawn_duration
            )

        else:

            self.yawn_start = None
            self.yawning = False
            yawn_duration = 0.0

        return {
            "drowsiness_alert":
                self.drowsiness_alert,

            "yawning":
                self.yawning,

            "closed_duration":
                closed_duration,

            "open_duration":
                (
                    open_duration
                    if not eyes_closed
                    else 0.0
                ),

            "yawn_duration":
                yawn_duration
        }