import cv2
import numpy as np
import tensorflow as tf


class EyeCNNPredictor:
    """
    Predicts whether eyes are open or closed.

    Model labels:
        0 = Closed
        1 = Open

    Supports batched prediction so that the left and right
    eye are processed in one CNN inference call.
    """

    def __init__(self, model_path):

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    def preprocess(self, eye_image):

        if eye_image is None or eye_image.size == 0:
            return None

        image = cv2.resize(
            eye_image,
            (224, 224)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        )

        return image

    def predict_batch(self, eye_images):

        processed_images = []

        for eye_image in eye_images:

            processed = self.preprocess(
                eye_image
            )

            if processed is None:
                return None

            processed_images.append(
                processed
            )

        batch = np.stack(
            processed_images,
            axis=0
        )

        probabilities = self.model.predict(
            batch,
            verbose=0
        ).reshape(-1)

        predictions = []

        for probability_open in probabilities:

            probability_open = float(
                probability_open
            )

            probability_closed = (
                1.0 - probability_open
            )

            predictions.append({
                "open_probability":
                    probability_open,

                "closed_probability":
                    probability_closed,

                "is_open":
                    probability_open >= 0.5
            })

        return predictions

    def predict(self, eye_image):

        result = self.predict_batch(
            [eye_image]
        )

        if result is None:
            return None

        return result[0]


class YawnCNNPredictor:
    """
    Predicts whether the mouth is yawning.

    Model labels:
        0 = No Yawn
        1 = Yawn
    """

    def __init__(self, model_path):

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    def preprocess(self, mouth_image):

        if mouth_image is None or mouth_image.size == 0:
            return None

        image = cv2.resize(
            mouth_image,
            (224, 224)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        )

        return image

    def predict(self, mouth_image):

        image = self.preprocess(
            mouth_image
        )

        if image is None:
            return None

        image = np.expand_dims(
            image,
            axis=0
        )

        probability_yawn = float(
            self.model.predict(
                image,
                verbose=0
            )[0][0]
        )

        probability_no_yawn = (
            1.0 - probability_yawn
        )

        return {
            "yawn_probability":
                probability_yawn,

            "no_yawn_probability":
                probability_no_yawn,

            "is_yawning":
                probability_yawn >= 0.5
        }

