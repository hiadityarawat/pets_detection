"""
=============================================================================
PET CLASSIFICATION TESTING SCRIPT (Cat vs Dog)
=============================================================================
Classify a pet photo as Cat or Dog and estimate the breed of a dog.
The first breed prediction needs internet access to download the
MobileNetV2 ImageNet weights and class labels.

Usage (PowerShell / Command Prompt):
    py test_pet.py path/to/image.jpg
    py test_pet.py --image path/to/image.jpg
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path

# Suppress TensorFlow informational/warning messages for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf


# =============================================================================
# CONSTANTS & CONFIGURATION (Matching obj.py)
# =============================================================================
MODEL_PATH = Path(__file__).resolve().with_name("cat_dog_cnn.keras")

# Image dimensions used during model training in obj.py
IMAGE_SIZE = 128

# Valid file extensions supported by the classifier
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Label mapping used during training:
# In obj.py:
#   cat_paths -> label 0
#   dog_paths -> label 1
# Dense(1, activation='sigmoid') produces probability of class 1 (Dog)
CLASS_NAMES = {0: "Cat", 1: "Dog"}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_trained_model(model_path: str = MODEL_PATH):
    """
    Loads the pre-trained Keras model from disk.
    Exits gracefully if the model file is missing or corrupted.
    """
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file not found: '{model_path}'")
        print("Keep 'cat_dog_cnn.keras' next to this script or pass --model.")
        sys.exit(1)

    try:
        print(f"[*] Loading model from '{model_path}'...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[+] Model loaded successfully.\n")
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed to load model '{model_path}': {e}")
        sys.exit(1)


def validate_image_path(image_path: str) -> Path:
    """
    Validates that the provided path exists and has a supported image extension.
    """
    path_obj = Path(image_path)

    # 1. Check if file exists
    if not path_obj.exists():
        print(f"\n[ERROR] File does not exist: '{image_path}'")
        sys.exit(1)

    # 2. Check if it is a file (not a directory)
    if not path_obj.is_file():
        print(f"\n[ERROR] Path is a directory, not an image file: '{image_path}'")
        sys.exit(1)

    # 3. Check for supported extension
    if path_obj.suffix.lower() not in VALID_EXTENSIONS:
        print(f"\n[ERROR] Unsupported image format: '{path_obj.suffix}'")
        print(f"Supported formats: {', '.join(sorted(VALID_EXTENSIONS))}")
        sys.exit(1)

    return path_obj


def preprocess_image(image_bgr: np.ndarray):
    """
    Preprocess image to match obj.py training pipeline:
      1. Convert BGR (OpenCV format) to RGB.
      2. Resize to (128, 128).
      3. Scale pixel values from [0, 255] to [0.0, 1.0].
      4. Add batch dimension: shape becomes (1, 128, 128, 3).
    """
    # Convert BGR to RGB (TensorFlow/Keras models expect RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize image to match training input size (128x128)
    resized_rgb = tf.image.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    return resized_rgb[tf.newaxis, ...] / 255.0


def predict_dog_breed(image_bgr):
    """Return the best ImageNet label when it is a domestic dog breed."""
    from tensorflow.keras.applications import MobileNetV2, mobilenet_v2

    model = MobileNetV2(weights="imagenet")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(image_rgb, (224, 224))
    batch = mobilenet_v2.preprocess_input(resized[tf.newaxis, ...])
    scores = model(batch, training=False).numpy()

    # ImageNet's zero-based domestic-dog classes run from 151 through 268.
    if not 151 <= int(np.argmax(scores[0])) <= 268:
        return None

    _, name, score = mobilenet_v2.decode_predictions(scores, top=1)[0][0]
    return name.replace("_", " "), float(score)


def predict(model, input_batch):
    """
    Runs model inference and interprets the sigmoid output:
      - Raw prediction is probability of class 1 (Dog).
      - If score >= 0.5: predicted class is Dog, confidence = score.
      - If score < 0.5:  predicted class is Cat, confidence = 1 - score.
    """
    raw_pred = float(model(input_batch, training=False)[0, 0])

    if raw_pred >= 0.5:
        label = "Dog"
        confidence = raw_pred
    else:
        label = "Cat"
        confidence = 1.0 - raw_pred

    confidence_pct = confidence * 100.0
    return label, confidence_pct, raw_pred


def display_result_cv2(image_bgr: np.ndarray, label: str, confidence_pct: float, image_name: str, wait_time: int = 0):
    """
    Displays the image in an OpenCV window with an annotated banner
    showing the prediction and confidence percentage.
    """
    # Make a copy so original image isn't modified
    display_img = image_bgr.copy()

    # Resize for viewing if the image is too large or too small
    h, w = display_img.shape[:2]
    max_dim = 800
    min_dim = 300
    scale = 1.0

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
    elif min(h, w) < min_dim:
        scale = min_dim / min(h, w)

    if scale != 1.0:
        display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    disp_h, disp_w = display_img.shape[:2]

    # Create top overlay banner for text readability
    banner_height = max(50, int(disp_h * 0.12))
    banner = np.zeros((banner_height, disp_w, 3), dtype=np.uint8)

    # Banner color: Warm Dark Red for Cat, Dark Blue for Dog
    banner_color = (40, 20, 90) if label == "Cat" else (90, 45, 20)
    banner[:] = banner_color

    # Annotation text
    text = f"{label}: {confidence_pct:.2f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.6, disp_w / 600.0)
    thickness = 2
    text_color = (0, 255, 128) if label == "Dog" else (0, 220, 255)

    # Position text vertically centered in banner
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = 15
    text_y = int((banner_height + text_size[1]) / 2)

    cv2.putText(banner, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Sub-text (instructions)
    sub_text = "Press any key to close"
    sub_font_scale = font_scale * 0.5
    sub_size = cv2.getTextSize(sub_text, font, sub_font_scale, 1)[0]
    sub_x = disp_w - sub_size[0] - 15
    cv2.putText(banner, sub_text, (sub_x, text_y), font, sub_font_scale, (200, 200, 200), 1, cv2.LINE_AA)

    # Combine banner and image vertically
    combined = np.vstack([banner, display_img])

    window_title = f"Pet Classifier - {label} ({confidence_pct:.1f}%) [{image_name}]"

    try:
        cv2.imshow(window_title, combined)
        print("\n[INFO] Displaying image window. Press any key on the image window to close it...")
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n[NOTE] Could not open GUI display window ({e}).")
        print("Terminal prediction output was completed successfully.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Classify a pet photo and estimate the breed if it is a dog."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the image to classify (e.g. 'pets_dataset/Cat/0.jpg')."
    )
    parser.add_argument(
        "--image", "-i",
        dest="flag_image_path",
        default=None,
        help="Alternative flag to specify image path."
    )
    parser.add_argument(
        "--model", "-m",
        dest="model_path",
        default=MODEL_PATH,
        help=f"Path to the trained .keras model (default: '{MODEL_PATH}')."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the OpenCV display window (terminal output only)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="OpenCV window timeout in milliseconds (default: 0 = wait until key pressed)."
    )

    args = parser.parse_args()

    # Determine image path from positional argument, flag argument, or user prompt
    input_path = args.image_path or args.flag_image_path
    if not input_path:
        print("=" * 60)
        print("CAT VS DOG IMAGE CLASSIFIER - TESTING TOOL")
        print("=" * 60)
        input_path = input("Enter path to pet image (e.g., test.jpg): ").strip(' "\'')

    if not input_path:
        print("\n[ERROR] No image path provided. Exiting.")
        sys.exit(1)

    # 1. Validate image path and format
    image_file = validate_image_path(input_path)

    # 2. Read image using OpenCV
    try:
        image_bgr = cv2.imdecode(np.frombuffer(image_file.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        image_bgr = None
    if image_bgr is None:
        print(f"\n[ERROR] OpenCV could not read/decode image: '{image_file}'")
        print("The file may be corrupted, empty, or an unsupported image format.")
        sys.exit(1)

    # 3. Load pre-trained model (do not retrain)
    model = load_trained_model(args.model_path)

    # 4. Preprocess image exactly as done during training (128x128, RGB, normalized [0,1])
    input_tensor = preprocess_image(image_bgr)

    # 5. Predict Cat vs Dog
    label, confidence_pct, raw_score = predict(model, input_tensor)

    # 6. Display results in terminal
    print("=" * 50)
    print("           PET CLASSIFICATION RESULT           ")
    print("=" * 50)
    print(f" Image File  : {image_file.name}")
    print(f" Image Path  : {image_file.resolve()}")
    print(f" Prediction  : {label}")
    print(f" Confidence  : {confidence_pct:.2f}%")
    print(f" Dog Score   : {raw_score:.4f}")
    if label == "Dog":
        try:
            breed = predict_dog_breed(image_bgr)
        except Exception as error:
            print(f" Breed       : unavailable ({error})")
            print(" Check internet access on first use so Keras can download the breed model and labels.")
        else:
            if breed is None:
                print(" Breed       : unknown")
            else:
                name, score = breed
                print(f" Likely Breed: {name} (model score {score:.1%})")
    print("=" * 50)

    # 7. Display image window with OpenCV (unless --no-gui is specified)
    if not args.no_gui:
        display_result_cv2(image_bgr, label, confidence_pct, image_file.name, wait_time=args.timeout)
    else:
        print("\n[INFO] GUI display skipped (--no-gui flag enabled).")


if __name__ == "__main__":
    main()
