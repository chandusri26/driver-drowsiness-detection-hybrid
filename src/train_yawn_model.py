"""
Yawn Detection using MobileNetV2
--------------------------------

Classes:
    0 = No Yawn
    1 = Yawn

Expected project structure:

driver drowsiness detction hybrid/
│
├── mouth/
│   ├── no yawn/
│   └── yawn/
│
├── models/
├── results/
├── src/
│   └── train_yawn_model.py
│
└── shape_predictor_68_face_landmarks.dat
"""

from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
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

ROOT_DIR = Path(__file__).resolve().parent.parent

# Actual folders in your project
NO_YAWN_DIR = ROOT_DIR / "mouth" / "no yawn"
YAWN_DIR = ROOT_DIR / "mouth" / "yawn"

MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 10
LEARNING_RATE = 1e-4

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


# ============================================================
# 2. PROJECT / DATASET CHECK
# ============================================================

print("=" * 70)
print("DRIVER DROWSINESS DETECTION - YAWN CNN TRAINING")
print("=" * 70)

print("\nProject root:")
print(ROOT_DIR)

if not NO_YAWN_DIR.exists():
    raise FileNotFoundError(
        f"No-yawn folder not found:\n{NO_YAWN_DIR}"
    )

if not YAWN_DIR.exists():
    raise FileNotFoundError(
        f"Yawn folder not found:\n{YAWN_DIR}"
    )

print("\n✓ no yawn folder found")
print("✓ yawn folder found")


# ============================================================
# 3. COLLECT IMAGES
# ============================================================

no_yawn_images = [
    p for p in NO_YAWN_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in VALID_EXTENSIONS
]

yawn_images = [
    p for p in YAWN_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in VALID_EXTENSIONS
]

print("\nDataset statistics")
print("-" * 70)

print(
    f"No-yawn images : {len(no_yawn_images):,}"
)

print(
    f"Yawn images    : {len(yawn_images):,}"
)

print(
    f"Total images   : "
    f"{len(no_yawn_images) + len(yawn_images):,}"
)

if len(no_yawn_images) == 0:
    raise RuntimeError(
        "No images found in the 'no yawn' folder."
    )

if len(yawn_images) == 0:
    raise RuntimeError(
        "No images found in the 'yawn' folder."
    )


# ============================================================
# 4. CREATE DATAFRAME
# ============================================================

records = []

# Label 0 = No Yawn
for path in no_yawn_images:
    records.append(
        {
            "path": str(path),
            "label": 0,
            "class_name": "No Yawn",
        }
    )

# Label 1 = Yawn
for path in yawn_images:
    records.append(
        {
            "path": str(path),
            "label": 1,
            "class_name": "Yawn",
        }
    )

df = pd.DataFrame(records)

print("\nClass distribution")
print("-" * 70)

print(
    df["class_name"].value_counts()
)


# ============================================================
# 5. OPTIONAL SUBJECT DETECTION
# ============================================================

def extract_subject_id(file_path):
    """
    Attempts to detect common subject-ID patterns.

    Examples:
        s0001_image.jpg -> s0001
        subject01_x.jpg -> subject01

    If no recognizable ID is found, return None.
    """

    filename = Path(file_path).name

    patterns = [
        r"^(s\d+)_",
        r"^(subject[_-]?\d+)",
        r"^(person[_-]?\d+)",
        r"^(p\d+)_",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            filename,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(1).lower()

    return None


df["subject"] = df["path"].apply(
    extract_subject_id
)

subject_count = df["subject"].notna().sum()

print("\nSubject-ID inspection")
print("-" * 70)

print(
    f"Images with recognizable subject IDs: "
    f"{subject_count:,} / {len(df):,}"
)


# ============================================================
# 6. DATASET SPLIT
# ============================================================

print("\nCreating train / validation / test split...")


# If meaningful subject IDs are present for ALL images,
# use subject-wise splitting.
#
# Otherwise use stratified image-level splitting.

use_subject_split = (
    df["subject"].notna().all()
    and df["subject"].nunique() >= 3
)


if use_subject_split:

    print(
        "✓ Subject IDs detected."
    )

    print(
        "Using subject-wise splitting."
    )

    from sklearn.model_selection import GroupShuffleSplit

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

    train_val_df = df.iloc[
        train_val_idx
    ].reset_index(drop=True)

    test_df = df.iloc[
        test_idx
    ].reset_index(drop=True)


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

    train_df = train_val_df.iloc[
        train_idx
    ].reset_index(drop=True)

    val_df = train_val_df.iloc[
        val_idx
    ].reset_index(drop=True)

    split_method = "subject-wise"


else:

    print(
        "No complete subject-ID structure detected."
    )

    print(
        "Using stratified image-level splitting."
    )

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=train_val_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_method = "stratified image-level"


# ============================================================
# 7. PRINT SPLIT INFORMATION
# ============================================================

print("\nSplit method:")
print(split_method)


def print_split_info(name, data):

    print(f"\n{name}")
    print("-" * 70)

    print(
        f"Images: {len(data):,}"
    )

    print(
        f"No Yawn: {(data['label'] == 0).sum():,}"
    )

    print(
        f"Yawn: {(data['label'] == 1).sum():,}"
    )


print_split_info(
    "TRAINING SET",
    train_df
)

print_split_info(
    "VALIDATION SET",
    val_df
)

print_split_info(
    "TEST SET",
    test_df
)


# ============================================================
# 8. VERIFY SUBJECT LEAKAGE IF APPLICABLE
# ============================================================

if use_subject_split:

    train_subjects = set(
        train_df["subject"]
    )

    val_subjects = set(
        val_df["subject"]
    )

    test_subjects = set(
        test_df["subject"]
    )

    print("\nSubject leakage check")
    print("-" * 70)

    print(
        "Train ∩ Validation:",
        len(
            train_subjects.intersection(
                val_subjects
            )
        )
    )

    print(
        "Train ∩ Test:",
        len(
            train_subjects.intersection(
                test_subjects
            )
        )
    )

    print(
        "Validation ∩ Test:",
        len(
            val_subjects.intersection(
                test_subjects
            )
        )
    )


# ============================================================
# 9. SAVE DATASET SPLITS
# ============================================================

train_df.to_csv(
    RESULTS_DIR / "yawn_train_split.csv",
    index=False
)

val_df.to_csv(
    RESULTS_DIR / "yawn_validation_split.csv",
    index=False
)

test_df.to_csv(
    RESULTS_DIR / "yawn_test_split.csv",
    index=False
)

print(
    "\n✓ Dataset split files saved."
)


# ============================================================
# 10. IMAGE LOADING
# ============================================================

def load_image(path, label):

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

    return (
        image,
        tf.cast(
            label,
            tf.float32
        )
    )


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
    name="yawn_data_augmentation"
)


# ============================================================
# 12. CREATE TF.DATA DATASETS
# ============================================================

def create_dataset(
    dataframe,
    training=False
):

    paths = dataframe[
        "path"
    ].values

    labels = dataframe[
        "label"
    ].values.astype(
        np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
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
                data_augmentation(
                    image,
                    training=True
                ),
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

print(
    "✓ TensorFlow datasets created."
)


# ============================================================
# 13. CLASS WEIGHTS
# ============================================================

no_yawn_count = (
    train_df["label"] == 0
).sum()

yawn_count = (
    train_df["label"] == 1
).sum()

total_count = (
    no_yawn_count +
    yawn_count
)

class_weight = {

    0:
        total_count /
        (2 * no_yawn_count),

    1:
        total_count /
        (2 * yawn_count),
}


print("\nClass weights")
print("-" * 70)

print(
    f"No Yawn: {class_weight[0]:.4f}"
)

print(
    f"Yawn   : {class_weight[1]:.4f}"
)


# ============================================================
# 14. HARDWARE CHECK
# ============================================================

print("\nHardware")
print("-" * 70)

gpus = tf.config.list_physical_devices(
    "GPU"
)

if gpus:

    print(
        f"GPU detected: {len(gpus)}"
    )

    for gpu in gpus:
        print(gpu)

else:

    print(
        "No GPU detected."
    )

    print(
        "Training will use CPU."
    )


# ============================================================
# 15. BUILD MOBILENETV2
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

# Freeze pretrained feature extractor
base_model.trainable = False


inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    name="mouth_image"
)


# Convert pixel values:
# 0-255 -> -1 to +1

x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(inputs)


x = base_model(
    x,
    training=False
)


x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


x = tf.keras.layers.Dropout(
    0.30
)(x)


outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="yawn_state"
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="Yawn_MobileNetV2"
)


# ============================================================
# 16. COMPILE
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

    ],
)


print("\nModel summary:")

model.summary()


# ============================================================
# 17. CALLBACKS
# ============================================================

model_path = (
    MODEL_DIR /
    "yawn_mobilenetv2.keras"
)


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(
            model_path
        ),

        monitor="val_loss",

        save_best_only=True,

        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True,

        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-7,

        verbose=1,
    ),

]


# ============================================================
# 18. TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING YAWN CNN TRAINING")
print("=" * 70)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weight,

    callbacks=callbacks,

    verbose=1,
)


# ============================================================
# 19. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    RESULTS_DIR /
    "yawn_training_history.csv",
    index=False
)

print(
    "\n✓ Training history saved."
)


# ============================================================
# 20. LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best model..."
)

model = tf.keras.models.load_model(
    model_path
)

print(
    "✓ Best model loaded."
)


# ============================================================
# 21. TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATING ON TEST DATA")
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
# 22. PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)


y_true = test_df[
    "label"
].values


probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()


y_pred = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# 23. METRICS
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
print("FINAL YAWN CNN METRICS")
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
        "No Yawn",
        "Yawn"
    ],

    zero_division=0,
)


print(
    "\nClassification Report"
)

print(
    "-" * 70
)

print(report)


with open(
    RESULTS_DIR /
    "yawn_classification_report.txt",

    "w",

    encoding="utf-8",
) as file:

    file.write(report)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print(
    "\nConfusion Matrix"
)

print(
    "-" * 70
)

print(cm)


np.savetxt(

    RESULTS_DIR /
    "yawn_confusion_matrix.csv",

    cm,

    delimiter=",",

    fmt="%d",
)


# ============================================================
# 26. ACCURACY CURVE
# ============================================================

history_data = history.history


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history_data["accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history_data["val_accuracy"],
    label="Validation Accuracy"
)


plt.title(
    "Yawn CNN Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    RESULTS_DIR /
    "yawn_accuracy_curve.png",

    dpi=150,
)


plt.close()


# ============================================================
# 27. LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history_data["loss"],
    label="Training Loss"
)


plt.plot(
    history_data["val_loss"],
    label="Validation Loss"
)


plt.title(
    "Yawn CNN Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    RESULTS_DIR /
    "yawn_loss_curve.png",

    dpi=150,
)


plt.close()


# ============================================================
# 28. CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(cm)


plt.title(
    "Yawn CNN Confusion Matrix"
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


plt.xticks(
    [0, 1],
    ["No Yawn", "Yawn"]
)


plt.yticks(
    [0, 1],
    ["No Yawn", "Yawn"]
)


for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            cm[i, j],

            ha="center",

            va="center",
        )


plt.colorbar()

plt.tight_layout()


plt.savefig(

    RESULTS_DIR /
    "yawn_confusion_matrix.png",

    dpi=150,
)


plt.close()


# ============================================================
# 29. SAVE FINAL MODEL
# ============================================================

final_model_path = (

    MODEL_DIR /
    "yawn_mobilenetv2_final.keras"

)


model.save(
    final_model_path
)


# ============================================================
# 30. COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("YAWN CNN TRAINING COMPLETE")
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


print("\nGenerated files:")

print(
    "✓ yawn_mobilenetv2.keras"
)

print(
    "✓ yawn_mobilenetv2_final.keras"
)

print(
    "✓ yawn_train_split.csv"
)

print(
    "✓ yawn_validation_split.csv"
)

print(
    "✓ yawn_test_split.csv"
)

print(
    "✓ yawn_training_history.csv"
)

print(
    "✓ yawn_classification_report.txt"
)

print(
    "✓ yawn_confusion_matrix.csv"
)

print(
    "✓ yawn_accuracy_curve.png"
)

print(
    "✓ yawn_loss_curve.png"
)

print(
    "✓ yawn_confusion_matrix.png"
)

print(
    "\nYawn CNN pipeline finished successfully."
)

