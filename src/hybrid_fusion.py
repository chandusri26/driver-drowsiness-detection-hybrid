import numpy as np


class HybridFusion:

    def __init__(
        self,
        ear_threshold=0.25,
        mar_threshold=0.60,
        eye_fusion_threshold=0.60,
        yawn_fusion_threshold=0.60
    ):

        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold

        self.eye_fusion_threshold = (
            eye_fusion_threshold
        )

        self.yawn_fusion_threshold = (
            yawn_fusion_threshold
        )

    @staticmethod
    def clip(value):
        return float(
            np.clip(
                value,
                0.0,
                1.0
            )
        )

    def calculate_eye_fusion(
        self,
        ear,
        eye_closed_probability
    ):

        # Geometric closed-eye score
        #
        # EAR <= 0.20 -> strongly closed
        # EAR >= 0.30 -> strongly open

        ear_closed_score = self.clip(
            (0.30 - ear) / 0.10
        )

        cnn_closed_score = (
            eye_closed_probability
        )

        # Equal contribution
        fused_score = (
            0.5 * ear_closed_score
            +
            0.5 * cnn_closed_score
        )

        eyes_closed = (
            fused_score
            >= self.eye_fusion_threshold
        )

        return {
            "ear_closed_score":
                ear_closed_score,

            "cnn_closed_score":
                cnn_closed_score,

            "eye_fusion_score":
                fused_score,

            "eyes_closed":
                eyes_closed
        }

    def calculate_yawn_fusion(
        self,
        mar,
        yawn_probability
    ):

        # Geometric yawn score
        #
        # MAR <= 0.40 -> low
        # MAR >= 0.70 -> strong yawn

        mar_yawn_score = self.clip(
            (mar - 0.40) / 0.30
        )

        cnn_yawn_score = (
            yawn_probability
        )

        fused_score = (
            0.5 * mar_yawn_score
            +
            0.5 * cnn_yawn_score
        )

        yawning = (
            fused_score
            >= self.yawn_fusion_threshold
        )

        return {
            "mar_yawn_score":
                mar_yawn_score,

            "cnn_yawn_score":
                cnn_yawn_score,

            "yawn_fusion_score":
                fused_score,

            "yawning":
                yawning
        }