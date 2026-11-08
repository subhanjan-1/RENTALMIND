"""
Day 6 prototype: train a CNN on Fashion-MNIST.

Run from the RetailMind repository root:
    python -m src.vision.fashion_mnist_prototype

Fashion-MNIST is used to validate the CNN architecture quickly before
training on the real Fashion Product Images dataset.
"""
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # Add the single-channel dimension expected by Conv2D.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Fashion-MNIST test loss: {test_loss:.4f}")
    print(f"Fashion-MNIST test accuracy: {test_accuracy:.4f}")

    model_path = MODEL_DIR / "fashion_mnist_prototype.keras"
    model.save(model_path)
    print(f"Saved prototype model: {model_path}")

    predictions = tf.argmax(model.predict(x_test, verbose=0), axis=1)
    matrix = tf.math.confusion_matrix(y_test, predictions, num_classes=10)
    print("Fashion-MNIST confusion matrix (rows=true, columns=predicted):")
    print(matrix.numpy())
    print("Class order:", CLASS_NAMES)

if __name__ == "__main__":
    main()
