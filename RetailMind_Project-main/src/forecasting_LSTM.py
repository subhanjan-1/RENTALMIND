# Day 9 - Rossmann Store Sales Forecasting (LSTM)
# Run from project root: python -u src/forecasting_LSTM.py
# Install: python -m pip install pandas numpy tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "rossmann-store-sales" / "train.csv"
MODEL_PATH = ROOT / "models" / "day9_lstm.keras"

STORE_ID = 1
LOOK_BACK = 28
TEST_DAYS = 30
EPOCHS = 50

# Load one store's data. Customers is deliberately excluded because it is
# generally unknown when forecasting future sales.
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
required = {"Store", "Date", "Sales", "Promo", "StateHoliday", "SchoolHoliday"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in train.csv: {sorted(missing)}")

store = df[df["Store"] == STORE_ID].copy()
if store.empty:
    raise ValueError(f"Store {STORE_ID} was not found.")

store["Date"] = pd.to_datetime(store["Date"])
store = store.sort_values("Date").set_index("Date")

# Keep useful features known from the calendar/business plan.
daily = store[["Sales", "Promo", "StateHoliday", "SchoolHoliday"]].asfreq("D")
daily["Sales"] = daily["Sales"].fillna(0)
daily["Promo"] = daily["Promo"].fillna(0)
daily["SchoolHoliday"] = daily["SchoolHoliday"].fillna(0)
daily["StateHoliday"] = daily["StateHoliday"].fillna("0").astype(str)
daily["StateHoliday"] = daily["StateHoliday"].map(
    {"0": 0, "a": 1, "b": 2, "c": 3}
).fillna(0)

if "Open" in store.columns:
    daily["Open"] = store["Open"].reindex(daily.index).fillna(0)
else:
    daily["Open"] = (daily["Sales"] > 0).astype(int)

# Sales are log-transformed to reduce the effect of very large sales days.
daily["sales_log"] = np.log1p(daily["Sales"].clip(lower=0))
day = daily.index.dayofweek
daily["dow_sin"] = np.sin(2 * np.pi * day / 7)
daily["dow_cos"] = np.cos(2 * np.pi * day / 7)

feature_cols = [
    "sales_log", "Promo", "StateHoliday", "SchoolHoliday",
    "Open", "dow_sin", "dow_cos"
]
features = daily[feature_cols].to_numpy(dtype="float32")
target = daily["sales_log"].to_numpy(dtype="float32")

if len(daily) <= LOOK_BACK + TEST_DAYS:
    raise ValueError("Not enough data for the selected look-back and test period.")

split = len(daily) - TEST_DAYS

# Standardize using training data only (avoids test-data leakage).
x_mean = features[:split].mean(axis=0)
x_std = features[:split].std(axis=0) + 1e-6
features = (features - x_mean) / x_std

y_mean = target[:split].mean()
y_std = target[:split].std() + 1e-6
target_scaled = (target - y_mean) / y_std

# Create chronological sequences.
X, y, target_indices = [], [], []
for i in range(LOOK_BACK, len(daily)):
    X.append(features[i - LOOK_BACK:i])
    y.append(target_scaled[i])
    target_indices.append(i)

X = np.asarray(X, dtype="float32")
y = np.asarray(y, dtype="float32")
target_indices = np.asarray(target_indices)

train_mask = target_indices < split
test_mask = target_indices >= split
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# LSTM with dropout and early stopping.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LOOK_BACK, len(feature_cols))),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber(),
    metrics=["mae"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
    )
]

model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=32,
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)

# Convert predictions back to sales units.
pred_scaled = model.predict(X_test, verbose=0).ravel()
pred_log = pred_scaled * y_std + y_mean
predicted = np.maximum(0, np.expm1(pred_log))
actual = daily["Sales"].to_numpy()[target_indices[test_mask]]

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
nonzero = actual != 0
mape = (
    np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    if nonzero.any() else float("nan")
)

# Weekly seasonal-naive baseline: use sales from 7 days earlier.
baseline = daily["Sales"].to_numpy()[target_indices[test_mask] - 7]
baseline_mae = np.mean(np.abs(actual - baseline))
baseline_rmse = np.sqrt(np.mean((actual - baseline) ** 2))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)

print(f"\nStore: {STORE_ID}")
print(f"LSTM  MAE:  {mae:,.2f}")
print(f"LSTM  RMSE: {rmse:,.2f}")
print(f"LSTM  MAPE: {mape:.2f}%")
print(f"7-day baseline MAE:  {baseline_mae:,.2f}")
print(f"7-day baseline RMSE: {baseline_rmse:,.2f}")
print(f"Model saved: {MODEL_PATH}")
