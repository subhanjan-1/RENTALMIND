from pathlib import Path
import csv
import json
import random

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "fashion-product-images-small"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "styles.csv"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


def load_records():
    """Load image paths and masterCategory labels."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset CSV: {CSV_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Could not find image directory: {IMAGE_DIR}"
        )

    records = []

    with CSV_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:

        reader = csv.DictReader(file)

        required = {"id", "masterCategory"}

        if not reader.fieldnames or not required.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "styles.csv must contain id and masterCategory columns. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:

            image_id = (row.get("id") or "").strip()
            category = (row.get("masterCategory") or "").strip()

            if not image_id or not category:
                continue

            image_path = IMAGE_DIR / f"{image_id}.jpg"

            if image_path.is_file():
                records.append(
                    (str(image_path), category)
                )

    if not records:
        raise ValueError(
            "No matching image files found."
        )

    return records


def stratified_split(records):
    """Create stratified train, validation and test datasets."""

    grouped = {}

    for path, category in records:
        grouped.setdefault(category, []).append(path)

    kept = {
        category: paths
        for category, paths in grouped.items()
        if len(paths) >= 3
    }

    dropped = len(grouped) - len(kept)

    if dropped:
        print(
            f"Skipping {dropped} categories "
            "with fewer than 3 images."
        )

    class_names = sorted(kept)

    label_to_id = {
        name: index
        for index, name in enumerate(class_names)
    }

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    rng = random.Random(SEED)

    for category in class_names:

        paths = kept[category][:]
        rng.shuffle(paths)

        n = len(paths)

        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))

        while n_test + n_val >= n:

            if n_test >= n_val and n_test > 1:
                n_test -= 1

            elif n_val > 1:
                n_val -= 1

            else:
                break

        test_part = paths[:n_test]

        val_part = paths[
            n_test:n_test + n_val
        ]

        train_part = paths[
            n_test + n_val:
        ]

        train_paths.extend(train_part)
        train_labels.extend(
            [label_to_id[category]] * len(train_part)
        )

        val_paths.extend(val_part)
        val_labels.extend(
            [label_to_id[category]] * len(val_part)
        )

        test_paths.extend(test_part)
        test_labels.extend(
            [label_to_id[category]] * len(test_part)
        )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
        class_names,
    )


def decode_and_resize(path, label):

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


def make_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 5000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def build_model(number_of_classes):

    augmentation = tf.keras.Sequential([

        tf.keras.layers.RandomFlip("horizontal"),

        tf.keras.layers.RandomRotation(0.05),

        tf.keras.layers.RandomZoom(0.10),

    ], name="category_image_augmentation")

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(*IMAGE_SIZE, 3)
        ),

        tf.keras.layers.Rescaling(1.0 / 255),

        augmentation,

        tf.keras.layers.Conv2D(
            32, 3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Conv2D(
            64, 3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(
            128, 3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.40),

        tf.keras.layers.Dense(
            number_of_classes,
            activation="softmax"
        ),

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


def main():

    print("\nLoading Fashion Product Dataset...")

    records = load_records()

    print(f"Usable images: {len(records)}")

    (
        train_data,
        val_data,
        test_data,
        class_names
    ) = stratified_split(records)

    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data

    print(f"\nMaster Categories: {len(class_names)}")

    print("Category names:", class_names)

    print(
        f"Train: {len(train_paths)} | "
        f"Validation: {len(val_paths)} | "
        f"Test: {len(test_paths)}"
    )

    train_ds = make_dataset(
        train_paths,
        train_labels,
        training=True
    )

    val_ds = make_dataset(
        val_paths,
        val_labels
    )

    test_ds = make_dataset(
        test_paths,
        test_labels
    )

    model = build_model(
        len(class_names)
    )

    model.summary()

    best_model_path = (
        MODEL_DIR / "vision_master_category_classifier.keras"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True
        ),

    ]

    print("\nTraining Master Category CNN...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model = tf.keras.models.load_model(
        best_model_path
    )

    print("\nEvaluating Category Classifier...")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    print(f"Category test loss: {test_loss:.4f}")

    print(
        f"Category test accuracy: "
        f"{test_accuracy:.4f}"
    )

    probabilities = model.predict(
        test_ds,
        verbose=1
    )

    y_pred = tf.argmax(
        probabilities,
        axis=1,
        output_type=tf.int32
    )

    y_true = tf.constant(
        test_labels,
        dtype=tf.int32
    )

    matrix = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(class_names)
    ).numpy()

    matrix_path = (
        MODEL_DIR / "vision_master_category_confusion_matrix.csv"
    )

    with matrix_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["true/predicted"] + class_names
        )

        for category, row in zip(
            class_names,
            matrix
        ):

            writer.writerow(
                [category] + row.tolist()
            )

    class_names_path = (
        MODEL_DIR / "vision_master_category_class_names.json"
    )

    with class_names_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=2
        )

    metrics_path = (
        MODEL_DIR / "vision_master_category_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "target": "masterCategory",
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "number_of_classes": len(class_names),
                "image_size": list(IMAGE_SIZE),
                "epochs_configured": EPOCHS,
                "batch_size": BATCH_SIZE,
                "train_samples": len(train_paths),
                "validation_samples": len(val_paths),
                "test_samples": len(test_paths)
            },
            file,
            indent=2
        )

    print("\nCategory classification completed!")

    print(f"Saved model: {best_model_path}")

    print(f"Saved labels: {class_names_path}")

    print(f"Saved confusion matrix: {matrix_path}")

    print(f"Saved metrics: {metrics_path}")

    print("Confusion matrix shape:", matrix.shape)

    print("Rows = true category; columns = predicted category.")


if __name__ == "__main__":
    main()
