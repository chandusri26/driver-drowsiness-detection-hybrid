import time


class DrowsinessDetector:

    def __init__(
        self,
        ear_threshold=0.25,
        mar_threshold=0.60,
        eye_closed_duration=5.0,
        eye_open_duration=3.0,
        yawn_duration=1.0
    ):

        # ------------------------------------------
        # Thresholds
        # ------------------------------------------

        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold

        # Time requirements in seconds
        self.eye_closed_duration = eye_closed_duration
        self.eye_open_duration = eye_open_duration
        self.yawn_duration = yawn_duration


        # ------------------------------------------
        # Timers
        # ------------------------------------------

        self.eye_closed_start = None
        self.eye_open_start = None

        self.yawn_start = None


        # ------------------------------------------
        # States
        # ------------------------------------------

        self.drowsiness_alert = False
        self.yawning = False


    def update(self, ear, mar):

        current_time = time.monotonic()


        # ==================================================
        # EYE CLOSURE DETECTION
        # ==================================================

        if ear < self.ear_threshold:

            # Eyes are closed

            self.eye_open_start = None


            # Start closed-eye timer
            if self.eye_closed_start is None:

                self.eye_closed_start = current_time


            closed_duration = (
                current_time - self.eye_closed_start
            )


            # Trigger drowsiness after 5 seconds
            if closed_duration >= self.eye_closed_duration:

                self.drowsiness_alert = True


        else:

            # Eyes are open

            self.eye_closed_start = None


            # If alarm/alert is active,
            # start measuring continuous open time
            if self.drowsiness_alert:

                if self.eye_open_start is None:

                    self.eye_open_start = current_time


                open_duration = (
                    current_time - self.eye_open_start
                )


                # Stop alert after 3 continuous
                # seconds of open eyes
                if open_duration >= self.eye_open_duration:

                    self.drowsiness_alert = False
                    self.eye_open_start = None

            else:

                self.eye_open_start = None


        # ==================================================
        # YAWN DETECTION
        # ==================================================

        if mar > self.mar_threshold:

            if self.yawn_start is None:

                self.yawn_start = current_time


            yawn_duration = (
                current_time - self.yawn_start
            )


            if yawn_duration >= self.yawn_duration:

                self.yawning = True

        else:

            self.yawn_start = None
            self.yawning = False


        # ==================================================
        # DURATIONS
        # ==================================================

        closed_duration = 0.0
        open_duration = 0.0
        yawn_duration = 0.0


        if self.eye_closed_start is not None:

            closed_duration = (
                current_time - self.eye_closed_start
            )


        if self.eye_open_start is not None:

            open_duration = (
                current_time - self.eye_open_start
            )


        if self.yawn_start is not None:

            yawn_duration = (
                current_time - self.yawn_start
            )


        # ==================================================
        # RETURN RESULT
        # ==================================================

        return {

            "drowsy": self.drowsiness_alert,

            "yawning": self.yawning,

            "closed_duration": closed_duration,

            "open_duration": open_duration,

            "yawn_duration": yawn_duration
        }