"""
RetailMind AI — Day 7 Vision Module (single-file implementation)

Features
--------
1. ResNet50 ImageNet transfer learning for masterCategory classification.
2. Fine-tuning of the top ResNet50 layers.
3. Single-image / directory masterCategory prediction.
4. Day 6 CNN vs Day 7 ResNet50 comparison on the same deterministic test split.
5. Pretrained Ultralytics YOLO11n object detection with bounding-box export.
6. YOLO evaluation on Ultralytics COCO8 sample dataset.

Expected dataset:
    data/fashion-product-images-small/
        images/
        styles.csv

Required CSV columns: id, masterCategory

Run from the RetailMind repository root:
    python src/vision/transfer_learning_day7.py --help

Dependencies:
    python -m pip install tensorflow ultralytics

Notes:
- Fashion Product Images Small has no bounding-box annotations.
- YOLO11n below is pretrained on COCO; it is not fine-tuned for fashion.
- COCO8 evaluation metrics are not fashion-specific mAP.
- Day 6 comparison expects the Day 6 masterCategory model and class mapping.
"""

from pathlib import Path
import argparse
import csv
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "fashion-product-images-small"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "styles.csv"
MODEL_DIR = ROOT / "models"
RUN_DIR = ROOT / "runs" / "vision_day7"

RESNET_MODEL = MODEL_DIR / "vision_resnet50_classifier.keras"
RESNET_CLASSES = MODEL_DIR / "vision_resnet50_class_names.json"
RESNET_METRICS = MODEL_DIR / "vision_resnet50_metrics.json"

DAY6_MODEL = MODEL_DIR / "vision_master_category_classifier.keras"
DAY6_CLASSES = MODEL_DIR / "vision_master_category_class_names.json"

YOLO_WEIGHTS = MODEL_DIR / "vision_yolo11n.pt"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LAST_LAYERS = 30
SEED = 42
AUTOTUNE = None


def require_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in the active Python environment. "
            "Run: python -m pip install tensorflow"
        ) from exc


def require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: "
            "python -m pip install ultralytics"
        ) from exc


def load_records():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset CSV: {CSV_PATH}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Missing image directory: {IMAGE_DIR}")

    records = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "masterCategory"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "styles.csv must contain columns id and masterCategory. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            image_id = (row.get("id") or "").strip()
            category = (row.get("masterCategory") or "").strip()
            if not image_id or not category:
                continue
            image_path = IMAGE_DIR / f"{image_id}.jpg"
            if image_path.is_file():
                records.append((str(image_path), category))

    if not records:
        raise ValueError("No matching images found in images/*.jpg.")
    return records


def stratified_split(records):
    """Deterministic split matching the Day 6 category-classifier logic."""
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
        print(f"Skipping {dropped} categories with fewer than 3 images.")

    class_names = sorted(kept)
    label_to_id = {name: index for index, name in enumerate(class_names)}
    rng = random.Random(SEED)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

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
        val_part = paths[n_test:n_test + n_val]
        train_part = paths[n_test + n_val:]

        train_paths.extend(train_part)
        train_labels.extend([label_to_id[category]] * len(train_part))
        val_paths.extend(val_part)
        val_labels.extend([label_to_id[category]] * len(val_part))
        test_paths.extend(test_part)
        test_labels.extend([label_to_id[category]] * len(test_part))

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
        class_names,
    )


def make_tf_dataset(tf, paths, labels, image_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def decode(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, image_size)
        return tf.cast(image, tf.float32), label

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_resnet_model(tf, number_of_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="resnet_image_augmentation",
    )

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(
        number_of_classes, activation="softmax", name="master_category"
    )(x)

    return tf.keras.Model(inputs, outputs, name="retailmind_resnet50"), backbone


def train_classifier(args):
    print('[Day 7] Starting ResNet50 training...', flush=True)
    print('[Day 7] Loading TensorFlow (this may take a moment)...', flush=True)
    tf = require_tensorflow()
    print(f'[Day 7] TensorFlow {tf.__version__} loaded.', flush=True)
    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f'[Day 7] Dataset CSV: {CSV_PATH}', flush=True)
    print(f'[Day 7] Image folder: {IMAGE_DIR}', flush=True)
    print('[Day 7] Scanning image files...', flush=True)
    records = load_records()
    print(f'[Day 7] Found {len(records)} usable images. Preparing split...', flush=True)
    train_data, val_data, test_data, class_names = stratified_split(records)
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data

    print(f"Usable images: {len(records)}")
    print(f"masterCategory classes: {len(class_names)} — {class_names}")
    print(
        f"Train: {len(train_paths)} | Validation: {len(val_paths)} | "
        f"Test: {len(test_paths)}"
    )

    print('[Day 7] Building TensorFlow input pipelines...', flush=True)
    train_ds = make_tf_dataset(
        tf, train_paths, train_labels, IMAGE_SIZE, training=True
    )
    val_ds = make_tf_dataset(tf, val_paths, val_labels, IMAGE_SIZE)
    test_ds = make_tf_dataset(tf, test_paths, test_labels, IMAGE_SIZE)

    print('[Day 7] Building ResNet50 with ImageNet weights...', flush=True)
    model, backbone = build_resnet_model(tf, len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(RESNET_MODEL),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7
        ),
    ]

    print("\n[Day 7] PHASE 1: Train head with frozen ResNet50", flush=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks,
    )

    print("\n[Day 7] PHASE 2: Fine-tune the top ResNet50 layers", flush=True)
    backbone.trainable = True
    for layer in backbone.layers[:-args.unfreeze_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(RESNET_MODEL)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)

    save_json(RESNET_CLASSES, class_names)
    metrics = {
        "target": "masterCategory",
        "backbone": "ResNet50",
        "pretrained_weights": "ImageNet",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": round(float(test_accuracy) * 100, 2),
        "number_of_classes": len(class_names),
        "train_samples": len(train_paths),
        "validation_samples": len(val_paths),
        "test_samples": len(test_paths),
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "head_epochs_completed": len(head_history.history["loss"]),
        "fine_tune_epochs_completed": len(fine_history.history["loss"]),
        "unfrozen_top_layers": args.unfreeze_layers,
    }
    save_json(RESNET_METRICS, metrics)

    print(f"\nResNet50 test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Saved model: {RESNET_MODEL}")
    print(f"Saved class mapping: {RESNET_CLASSES}")
    print(f"Saved metrics: {RESNET_METRICS}")


def predict_classifier(args):
    tf = require_tensorflow()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not RESNET_MODEL.is_file() or not RESNET_CLASSES.is_file():
        raise FileNotFoundError(
            "Train the ResNet50 classifier first with "
            "'python src/vision/transfer_learning_day7.py train'."
        )

    model = tf.keras.models.load_model(RESNET_MODEL)
    with RESNET_CLASSES.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    image = tf.io.decode_image(
        tf.io.read_file(str(image_path)),
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)

    probabilities = model.predict(image, verbose=0)[0]
    top = probabilities.argsort()[::-1][: min(3, len(class_names))]

    print("\nRETAILMIND — RESNET50 MASTER CATEGORY")
    print(f"Image: {image_path}")
    print(f"Prediction: {class_names[int(top[0])]}")
    print(f"Confidence: {float(probabilities[top[0]]) * 100:.2f}%")
    print("Top predictions:")
    for rank, index in enumerate(top, 1):
        print(
            f"{rank}. {class_names[int(index)]}: "
            f"{float(probabilities[index]) * 100:.2f}%"
        )


def evaluate_comparison(args):
    tf = require_tensorflow()
    required = [DAY6_MODEL, DAY6_CLASSES, RESNET_MODEL, RESNET_CLASSES]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Train Day 6 category and Day 7 models first:\n"
            + "\n".join(missing)
        )

    (_, _, test_data, split_classes) = stratified_split(load_records())
    test_paths, test_labels = test_data

    with DAY6_CLASSES.open("r", encoding="utf-8") as f:
        day6_classes = json.load(f)
    with RESNET_CLASSES.open("r", encoding="utf-8") as f:
        day7_classes = json.load(f)

    if split_classes != day6_classes or split_classes != day7_classes:
        raise ValueError(
            "Class mappings do not match the current deterministic split. "
            "Retrain both category classifiers using the same split logic."
        )

    day6 = tf.keras.models.load_model(DAY6_MODEL)
    day7 = tf.keras.models.load_model(RESNET_MODEL)

    day6_ds = make_tf_dataset(
        tf, test_paths, test_labels, (128, 128)
    )
    day7_ds = make_tf_dataset(
        tf, test_paths, test_labels, IMAGE_SIZE
    )

    day6_loss, day6_acc = day6.evaluate(day6_ds, verbose=1)
    day7_loss, day7_acc = day7.evaluate(day7_ds, verbose=1)

    result = {
        "target": "masterCategory",
        "same_test_split": True,
        "test_samples": len(test_paths),
        "day6_cnn": {
            "test_loss": float(day6_loss),
            "test_accuracy": float(day6_acc),
            "test_accuracy_percent": round(float(day6_acc) * 100, 2),
        },
        "day7_resnet50": {
            "test_loss": float(day7_loss),
            "test_accuracy": float(day7_acc),
            "test_accuracy_percent": round(float(day7_acc) * 100, 2),
        },
        "accuracy_change_percentage_points": round(
            (float(day7_acc) - float(day6_acc)) * 100, 2
        ),
    }
    output = MODEL_DIR / "vision_day7_comparison.json"
    save_json(output, result)

    print("\nDAY 6 vs DAY 7 — MASTER CATEGORY")
    print(f"Day 6 CNN: {day6_acc * 100:.2f}%")
    print(f"Day 7 ResNet50: {day7_acc * 100:.2f}%")
    print(
        "Change: "
        f"{result['accuracy_change_percentage_points']:+.2f} percentage points"
    )
    print(f"Saved comparison: {output}")


def detect_yolo(args):
    YOLO = require_ultralytics()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Detection source not found: {source}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if args.weights:
        weights = Path(args.weights)
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        model = YOLO(str(weights))
    elif YOLO_WEIGHTS.is_file():
        model = YOLO(str(YOLO_WEIGHTS))
    else:
        model = YOLO("yolo11n.pt")
        model.save(str(YOLO_WEIGHTS))

    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(RUN_DIR),
        name="detections",
        exist_ok=True,
        verbose=True,
    )

    all_detections = []
    for result in results:
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [
                    float(v) for v in box.xyxy[0].tolist()
                ]
                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [
                        round(x1, 2), round(y1, 2),
                        round(x2, 2), round(y2, 2),
                    ],
                })
        all_detections.append({
            "image": str(result.path),
            "detection_count": len(detections),
            "detections": detections,
        })

    output_dir = RUN_DIR / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "detections.json"
    save_json(output_json, all_detections)

    print(f"Annotated images: {output_dir}")
    print(f"Bounding-box data: {output_json}")
    print(
        "Reminder: pretrained YOLO11n uses COCO object classes and is not "
        "a fashion-specific detector."
    )


def evaluate_yolo(args):
    YOLO = require_ultralytics()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(YOLO_WEIGHTS) if YOLO_WEIGHTS.is_file() else "yolo11n.pt")
    metrics = model.val(
        data="coco8.yaml",
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
        project=str(RUN_DIR),
        name="yolo_coco8_validation",
        exist_ok=True,
    )

    result = {
        "weights": "YOLO11n pretrained COCO weights",
        "validation_dataset": "COCO8 sample",
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    output = MODEL_DIR / "vision_yolo_coco8_metrics.json"
    save_json(output, result)

    print("\nYOLO COCO8 evaluation")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"Saved metrics: {output}")
    print("These are COCO8 metrics, not fashion-specific detection metrics.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="RetailMind Day 7: ResNet50 transfer learning + YOLO detection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train", help="Train ResNet50 head and fine-tune top layers"
    )
    train.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    train.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    train.add_argument("--unfreeze-layers", type=int, default=FINE_TUNE_LAST_LAYERS)
    train.set_defaults(func=train_classifier)

    predict = subparsers.add_parser(
        "predict", help="Predict masterCategory for one image"
    )
    predict.add_argument("image")
    predict.set_defaults(func=predict_classifier)

    compare = subparsers.add_parser(
        "compare", help="Compare Day 6 CNN and Day 7 ResNet50"
    )
    compare.set_defaults(func=evaluate_comparison)

    detect = subparsers.add_parser(
        "detect", help="Run pretrained YOLO object detection"
    )
    detect.add_argument("source", help="Image path or image directory")
    detect.add_argument("--weights", default=None)
    detect.add_argument("--conf", type=float, default=0.25)
    detect.add_argument("--imgsz", type=int, default=640)
    detect.set_defaults(func=detect_yolo)

    yolo_eval = subparsers.add_parser(
        "evaluate-yolo", help="Evaluate pretrained YOLO on COCO8"
    )
    yolo_eval.add_argument("--imgsz", type=int, default=640)
    yolo_eval.add_argument("--batch", type=int, default=8)
    yolo_eval.set_defaults(func=evaluate_yolo)

    return parser


def main():
    print('[Day 7] RetailMind vision script started.', flush=True)
    print(f'[Day 7] Project root: {ROOT}', flush=True)
    parser = build_parser()
    args = parser.parse_args()
    print(f'[Day 7] Selected command: {args.command}', flush=True)
    try:
        args.func(args)
    except Exception as exc:
        print(f'\n[Day 7] ERROR: {exc}', file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
