"""
Eye State Classification using MobileNetV2
-------------------------------------------
Classes:
    0 = Closed Eyes
    1 = Open Eyes

Dataset:
    MRL Eye Dataset

Important:
    The dataset is split by SUBJECT ID rather than individual images
    to reduce train/test data leakage.

Expected project structure:

driver drowsiness detction hybrid/
│
├── eye/
│   ├── Open-Eyes/
│   └── Close-Eyes/
│
├── models/
├── results/
├── src/
│   └── train_eye_model.py
│
└── shape_predictor_68_face_landmarks.dat
"""

from pathlib import Path
import re
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dataset directories
OPEN_DIR = ROOT_DIR / "eye" / "Open-Eyes"
CLOSED_DIR = ROOT_DIR / "eye" / "Close-Eyes"

# Output directories
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4

# Split ratios
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


# ============================================================
# 2. CHECK DATASET DIRECTORIES
# ============================================================

print("=" * 70)
print("DRIVER DROWSINESS DETECTION - EYE CNN TRAINING")
print("=" * 70)

print("\nProject root:")
print(ROOT_DIR)

print("\nChecking dataset directories...")

if not OPEN_DIR.exists():
    raise FileNotFoundError(
        f"Open-Eyes folder not found:\n{OPEN_DIR}"
    )

if not CLOSED_DIR.exists():
    raise FileNotFoundError(
        f"Close-Eyes folder not found:\n{CLOSED_DIR}"
    )

print("✓ Open-Eyes folder found")
print("✓ Close-Eyes folder found")


# ============================================================
# 3. COLLECT IMAGE PATHS
# ============================================================

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}

open_images = [
    p for p in OPEN_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
]

closed_images = [
    p for p in CLOSED_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
]

print("\nDataset statistics")
print("-" * 70)

print(f"Open-eye images   : {len(open_images):,}")
print(f"Closed-eye images : {len(closed_images):,}")
print(f"Total images      : {len(open_images) + len(closed_images):,}")

if len(open_images) == 0 or len(closed_images) == 0:
    raise RuntimeError(
        "One or both eye classes contain no images."
    )


# ============================================================
# 4. EXTRACT SUBJECT ID
# ============================================================

def extract_subject_id(file_path):
    """
    MRL filenames generally look like:

        s0001_01842_0_0_0_...
        s0002_...

    The subject ID is the part beginning with 's'.

    Example:
        s0001_01842_...png
        -> s0001
    """

    filename = file_path.name

    match = re.match(r"(s\d+)_", filename)

    if match:
        return match.group(1)

    # Fallback if filename doesn't follow expected MRL format
    return file_path.stem


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

records = []

for path in open_images:
    records.append(
        {
            "path": str(path),
            "label": 1,
            "class_name": "Open",
            "subject": extract_subject_id(path),
        }
    )

for path in closed_images:
    records.append(
        {
            "path": str(path),
            "label": 0,
            "class_name": "Closed",
            "subject": extract_subject_id(path),
        }
    )

df = pd.DataFrame(records)

print("\nSubject information")
print("-" * 70)

print(f"Unique subjects: {df['subject'].nunique():,}")


# ============================================================
# 6. SUBJECT-WISE TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating subject-wise dataset split...")

# First split:
# 80% development
# 20% test

splitter_test = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=SEED,
)

train_val_idx, test_idx = next(
    splitter_test.split(
        df,
        groups=df["subject"]
    )
)

train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)


# Second split:
# 80% train
# 20% validation

splitter_val = GroupShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
)

train_idx, val_idx = next(
    splitter_val.split(
        train_val_df,
        groups=train_val_df["subject"]
    )
)

train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
val_df = train_val_df.iloc[val_idx].reset_index(drop=True)


# ============================================================
# 7. VERIFY SUBJECT LEAKAGE
# ============================================================

train_subjects = set(train_df["subject"])
val_subjects = set(val_df["subject"])
test_subjects = set(test_df["subject"])

print("\nChecking subject leakage...")

print(
    "Train ∩ Validation:",
    len(train_subjects.intersection(val_subjects))
)

print(
    "Train ∩ Test:",
    len(train_subjects.intersection(test_subjects))
)

print(
    "Validation ∩ Test:",
    len(val_subjects.intersection(test_subjects))
)

if (
    train_subjects.intersection(val_subjects)
    or train_subjects.intersection(test_subjects)
    or val_subjects.intersection(test_subjects)
):
    raise RuntimeError(
        "Subject leakage detected!"
    )

print("✓ No subject leakage detected")


# ============================================================
# 8. DISPLAY SPLIT STATISTICS
# ============================================================

def print_split_info(name, data):
    print(f"\n{name}")
    print("-" * 70)

    print(f"Images   : {len(data):,}")
    print(f"Subjects : {data['subject'].nunique():,}")

    print(
        f"Closed   : {(data['label'] == 0).sum():,}"
    )

    print(
        f"Open     : {(data['label'] == 1).sum():,}"
    )


print_split_info("TRAINING SET", train_df)
print_split_info("VALIDATION SET", val_df)
print_split_info("TEST SET", test_df)


# ============================================================
# 9. SAVE DATASET SPLIT
# ============================================================

train_df.to_csv(
    RESULTS_DIR / "eye_train_split.csv",
    index=False
)

val_df.to_csv(
    RESULTS_DIR / "eye_validation_split.csv",
    index=False
)

test_df.to_csv(
    RESULTS_DIR / "eye_test_split.csv",
    index=False
)

print("\n✓ Dataset split information saved to results/")


# ============================================================
# 10. IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):
    """
    Load image and convert it to RGB.

    MRL images are generally grayscale.
    MobileNetV2 expects 3-channel input,
    so grayscale images are loaded as RGB.
    """

    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, tf.cast(label, tf.float32)


# ============================================================
# 11. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            factor=0.05
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomContrast(
            factor=0.10
        ),
    ],
    name="eye_data_augmentation"
)


# ============================================================
# 12. CREATE TF.DATA DATASETS
# ============================================================

def create_dataset(dataframe, training=False):

    paths = dataframe["path"].values

    labels = dataframe["label"].values.astype(
        np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(
                len(dataframe),
                10000
            ),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            lambda image, label: (
                data_augmentation(image, training=True),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_dataset = create_dataset(
    train_df,
    training=True
)

validation_dataset = create_dataset(
    val_df,
    training=False
)

test_dataset = create_dataset(
    test_df,
    training=False
)

print("\n✓ TensorFlow datasets created")


# ============================================================
# 13. CLASS WEIGHTS
# ============================================================

closed_count = (train_df["label"] == 0).sum()
open_count = (train_df["label"] == 1).sum()

total_count = closed_count + open_count

class_weight = {
    0: total_count / (2 * closed_count),
    1: total_count / (2 * open_count),
}

print("\nClass weights")
print("-" * 70)

print(
    f"Closed: {class_weight[0]:.4f}"
)

print(
    f"Open  : {class_weight[1]:.4f}"
)


# ============================================================
# 14. CHECK GPU
# ============================================================

print("\nHardware")
print("-" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"GPU detected: {len(gpus)}")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected.")
    print("Training will use CPU.")


# ============================================================
# 15. BUILD MOBILENETV2 MODEL
# ============================================================

print("\nBuilding MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False


inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    name="eye_image"
)


# Convert [0, 255] -> [-1, 1]
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="eye_state"
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="Eye_MobileNetV2"
)


# ============================================================
# 16. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),
    ]
)


print("\nModel summary:")
model.summary()


# ============================================================
# 17. CALLBACKS
# ============================================================

model_path = (
    MODEL_DIR /
    "eye_mobilenetv2.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
]


# ============================================================
# 18. TRAIN MODEL
# ============================================================

print("\n")
print("=" * 70)
print("STARTING EYE CNN TRAINING")
print("=" * 70)

history = model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weight,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# 19. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    RESULTS_DIR /
    "eye_training_history.csv",
    index=False
)

print(
    "\n✓ Training history saved."
)


# ============================================================
# 20. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    model_path
)

print("✓ Best model loaded")


# ============================================================
# 21. EVALUATE TEST DATA
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATING ON UNSEEN TEST SUBJECTS")
print("=" * 70)

test_results = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nKeras test metrics:")

for metric_name, value in zip(
    model.metrics_names,
    test_results
):
    print(
        f"{metric_name}: {value:.4f}"
    )


# ============================================================
# 22. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = test_df["label"].values

probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()

y_pred = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# 23. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

print("\n")
print("=" * 70)
print("FINAL EYE CNN METRICS")
print("=" * 70)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# 24. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Closed Eyes",
        "Open Eyes"
    ],
    zero_division=0
)

print("\nClassification Report")
print("-" * 70)

print(report)

with open(
    RESULTS_DIR /
    "eye_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix")
print("-" * 70)

print(cm)

np.savetxt(
    RESULTS_DIR /
    "eye_confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# 26. TRAINING CURVES
# ============================================================

history_data = history.history

# Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history_data["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_data["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Eye CNN Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "eye_accuracy_curve.png",
    dpi=150
)

plt.close()


# Loss
plt.figure(figsize=(8, 5))

plt.plot(
    history_data["loss"],
    label="Training Loss"
)

plt.plot(
    history_data["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Eye CNN Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "eye_loss_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# 27. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "Eye CNN Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    ["Closed", "Open"]
)

plt.yticks(
    [0, 1],
    ["Closed", "Open"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "eye_confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# 28. SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR /
    "eye_mobilenetv2_final.keras"
)

model.save(
    final_model_path
)

print("\n")
print("=" * 70)
print("EYE CNN TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest model:\n{model_path}"
)

print(
    f"\nFinal model:\n{final_model_path}"
)

print(
    f"\nResults directory:\n{RESULTS_DIR}"
)

print("\nGenerated files include:")

print("✓ eye_mobilenetv2.keras")
print("✓ eye_mobilenetv2_final.keras")
print("✓ eye_train_split.csv")
print("✓ eye_validation_split.csv")
print("✓ eye_test_split.csv")
print("✓ eye_training_history.csv")
print("✓ eye_classification_report.txt")
print("✓ eye_confusion_matrix.csv")
print("✓ eye_accuracy_curve.png")
print("✓ eye_loss_curve.png")
print("✓ eye_confusion_matrix.png")

print("\nEye CNN pipeline finished successfully.")

