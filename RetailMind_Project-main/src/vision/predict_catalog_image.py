"""
Predict the articleType category for one product image.

Run from the RetailMind repository root:
    python -m src.vision.predict_catalog_image data/fashion-product-images-small/images/15970.jpg
"""
from pathlib import Path
import json
import sys
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "vision_catalog_classifier.keras"
CLASS_NAMES_PATH = ROOT / "models" / "vision_class_names.json"
IMAGE_SIZE = (128, 128)

def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m src.vision.predict_catalog_image <image_path>"
        )

    image_path = Path(sys.argv[1])
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            "Train the catalog classifier first. Expected model and class "
            "labels in the models/ directory."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    image = tf.keras.utils.load_img(
        image_path, target_size=IMAGE_SIZE
    )
    array = tf.keras.utils.img_to_array(image)
    array = tf.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    best_index = int(tf.argmax(probabilities))
    print(f"Predicted category: {class_names[best_index]}")
    print(f"Confidence: {probabilities[best_index]:.2%}")

if __name__ == "__main__":
    main()
