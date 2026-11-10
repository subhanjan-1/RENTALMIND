

from pathlib import Path
import argparse
import json

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "models" / "vision_master_category_classifier.keras"
)

CLASS_NAMES_PATH = (
    ROOT / "models" / "vision_master_category_class_names.json"
)

IMAGE_SIZE = (128, 128)


def load_resources():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Train the model first using "
            "python -m src.vision.train_category_classifier"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with CLASS_NAMES_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output classes do not match class names."
        )

    return model, class_names


def preprocess_image(image_path):

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # Model already contains Rescaling(1.0 / 255).
    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


def predict_category(model, class_names, image_path):

    image = preprocess_image(image_path)

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    top_indices = probabilities.argsort()[::-1][:3]

    predicted_index = int(top_indices[0])

    predicted_category = class_names[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    print("\n" + "=" * 45)
    print("       RETAILMIND CATEGORY PREDICTION")
    print("=" * 45)

    print(f"\nImage: {image_path}")

    print(f"\nPredicted Master Category: {predicted_category}")

    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nTop 3 Predictions:")

    for rank, index in enumerate(top_indices, start=1):

        category = class_names[int(index)]

        score = float(probabilities[index]) * 100

        print(
            f"{rank}. {category}: {score:.2f}%"
        )

    print("\n" + "=" * 45)

    return predicted_category, confidence


def main():

    parser = argparse.ArgumentParser(
        description="Predict fashion product master category."
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the fashion product image."
    )

    args = parser.parse_args()

    model, class_names = load_resources()

    predict_category(
        model,
        class_names,
        args.image
    )


if __name__ == "__main__":
    main()
