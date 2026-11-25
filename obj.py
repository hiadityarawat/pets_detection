# ============================================================
# PET (CAT vs DOG) CLASSIFICATION + OBJECT DETECTION
# Memory-efficient version with corrupt-image filtering
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from pathlib import Path
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("pets_dataset")
CAT_PATH = DATASET_PATH / "Cat"
DOG_PATH = DATASET_PATH / "Dog"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
RANDOM_STATE = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# CHECK DATASET FOLDERS
# ============================================================

for folder in (DATASET_PATH, CAT_PATH, DOG_PATH):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")


# ============================================================
# COLLECT IMAGE PATHS (paths only, not images -> saves RAM)
# ============================================================

def get_image_paths(folder):
    paths = []
    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
            paths.append(str(file))
    return sorted(paths)


def is_valid_image(path):
    """Return True only if TensorFlow can fully decode the image
    into a 3-channel RGB array (same decoder used in training)."""
    try:
        data = tf.io.read_file(path)
        img = tf.image.decode_image(
            data, channels=3, expand_animations=False
        )
        return img.shape.rank == 3
    except Exception:
        return False


cat_paths = get_image_paths(CAT_PATH)
dog_paths = get_image_paths(DOG_PATH)

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")
print("Cat images found:", len(cat_paths))
print("Dog images found:", len(dog_paths))

print("\nChecking for corrupt images (this takes a few minutes)...")

cat_before, dog_before = len(cat_paths), len(dog_paths)
cat_paths = [p for p in cat_paths if is_valid_image(p)]
dog_paths = [p for p in dog_paths if is_valid_image(p)]

print("Removed corrupt cat images:", cat_before - len(cat_paths))
print("Removed corrupt dog images:", dog_before - len(dog_paths))
print("Valid cat images:", len(cat_paths))
print("Valid dog images:", len(dog_paths))


# ============================================================
# CREATE LABELS  (Cat = 0, Dog = 1)
# ============================================================

all_paths = np.array(cat_paths + dog_paths)
all_labels = np.array(
    [0] * len(cat_paths) + [1] * len(dog_paths), dtype=np.int32
)

print("Total valid images:", len(all_paths))


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)
# ============================================================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.30, random_state=RANDOM_STATE, stratify=all_labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels,
    test_size=0.50, random_state=RANDOM_STATE, stratify=temp_labels
)

print("\n========================================")
print("DATASET SPLIT")
print("========================================")
print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Testing images:", len(test_paths))


# ============================================================
# tf.data PIPELINE
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0   # 0-255 -> 0-1
    return image, label


def create_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 5000), seed=RANDOM_STATE
        )

    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


train_dataset = create_dataset(train_paths, train_labels, training=True)
validation_dataset = create_dataset(val_paths, val_labels)
test_dataset = create_dataset(test_paths, test_labels)


# ============================================================
# CNN MODEL (Sequential)
# ============================================================

print("\n========================================")
print("BUILDING CNN MODEL")
print("========================================")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    # Data augmentation (only active during training)
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    # Convolution blocks
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Classifier head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Binary output: 0 = Cat, 1 = Dog
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
)

model.save("cat_dog_cnn.keras")
print("\nModel saved as: cat_dog_cnn.keras")


# ============================================================
# EVALUATE
# ============================================================

print("\n========================================")
print("EVALUATING MODEL")
print("========================================")

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# ============================================================
# CLASSIFICATION REPORT + CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

predictions = model.predict(test_dataset)
predicted_labels = (predictions.flatten() >= 0.5).astype(int)

print(classification_report(
    test_labels, predicted_labels, target_names=["Cat", "Dog"]
))

cm = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)


# ============================================================
# TRAINING GRAPHS
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def classify_array(rgb_array):
    """rgb_array: HxWx3 image. Returns (label, confidence)."""
    tensor = tf.convert_to_tensor(rgb_array, dtype=tf.float32)
    tensor = tf.image.resize(tensor, [IMAGE_SIZE, IMAGE_SIZE]) / 255.0
    tensor = tf.expand_dims(tensor, axis=0)

    prediction = float(model.predict(tensor, verbose=0)[0][0])

    if prediction >= 0.5:
        return "Dog", prediction
    return "Cat", 1 - prediction


def predict_pet(image_path):
    image_path = str(image_path)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image, channels=3, expand_animations=False
    ).numpy()

    label, confidence = classify_array(image)

    print("\n========================================")
    print("PREDICTION")
    print("========================================")
    print("Image:", image_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")


# ============================================================
# OBJECT DETECTION (SSD MobileNet) + CLASSIFICATION
# ============================================================

# COCO class ids used by SSD MobileNet: 17 = cat, 18 = dog
COCO_PET_IDS = {17, 18}

detector = None

print("\n========================================")
print("LOADING SSD MOBILENET")
print("========================================")

try:
    import tensorflow_hub as hub
    detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    print("SSD MobileNet loaded successfully.")
except Exception as error:
    print("Could not load the object detector (needs internet and")
    print("the tensorflow-hub package). Detection will be skipped.")
    print("Reason:", error)


def detect_pet(image_path):
    if detector is None:
        print("Detector not available.")
        return

    import cv2

    image_path = str(image_path)

    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]

    input_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detector(input_tensor)

    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)

    # Select the best-scoring cat or dog detection.
    best_index = None
    best_score = 0.0

    for i, (score, cls) in enumerate(zip(scores, classes)):
        if cls in COCO_PET_IDS and score > best_score:
            best_score = score
            best_index = i

    if best_index is None or best_score < 0.30:
        print("No confident cat or dog detected.")
        return

    ymin, xmin, ymax, xmax = boxes[best_index]

    xmin = max(0, int(xmin * width))
    xmax = min(width, int(xmax * width))
    ymin = max(0, int(ymin * height))
    ymax = min(height, int(ymax * height))

    cropped = rgb_image[ymin:ymax, xmin:xmax]

    if cropped.size == 0:
        print("Invalid detected region.")
        return

    label, confidence = classify_array(cropped)

    output_image = rgb_image.copy()

    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    cv2.putText(
        output_image,
        f"{label} {confidence * 100:.1f}%",
        (xmin, max(ymin - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    plt.figure(figsize=(10, 7))
    plt.imshow(output_image)
    plt.title(f"Detected: {label} ({confidence * 100:.2f}%)")
    plt.axis("off")
    plt.show()

    print("\n========================================")
    print("OBJECT DETECTION RESULT")
    print("========================================")
    print("Object:", label)
    print(f"Classification confidence: {confidence * 100:.2f}%")
    print(f"Detection confidence: {best_score * 100:.2f}%")
    print("Bounding box:", (xmin, ymin, xmax, ymax))


# ============================================================
# TEST IMAGE (optional)
# ============================================================

TEST_IMAGE = "test.jpg"

if os.path.exists(TEST_IMAGE):
    predict_pet(TEST_IMAGE)
    detect_pet(TEST_IMAGE)
else:
    print(f"\nNo {TEST_IMAGE} found.")
    print("Training and evaluation are complete.")
