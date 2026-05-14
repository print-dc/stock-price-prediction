"""
cnn_model.py
1-D CNN for stock price time-series forecasting.

Architecture
------------
Input  : (batch, window_size, n_features)

Block 1  Conv1D(64, 3, ReLU) → BatchNorm → MaxPool → Dropout
Block 2  Conv1D(128, 3, ReLU) → BatchNorm → MaxPool → Dropout
Block 3  Conv1D(256, 3, ReLU) → BatchNorm → GlobalAvgPool

Dual heads
  ┌─ price_head : Dense(128) → Dense(64) → Dense(1, linear)   (regression)
  └─ trend_head : Dense(128) → Dense(64) → Dense(1, sigmoid)  (binary classification)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn_model(
    window_size: int,
    n_features: int,
    dropout_rate: float = 0.3,
    l2: float = 1e-4,
) -> keras.Model:
    """
    Build and compile a dual-head 1-D CNN model.

    Parameters
    ----------
    window_size  : lookback length (time steps)
    n_features   : number of input features per step
    dropout_rate : dropout probability applied after each conv block
    l2           : L2 regularisation weight

    Returns
    -------
    Compiled Keras Model with two outputs:
        'price_output' – next-day normalised close price
        'trend_output' – probability that price increases
    """
    reg = regularizers.l2(l2)
    inp = keras.Input(shape=(window_size, n_features), name="input")

    # ── Conv Block 1 ──────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu",
                      kernel_regularizer=reg, name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # ── Conv Block 2 ──────────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="causal", activation="relu",
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # ── Conv Block 3 ──────────────────────────────────
    x = layers.Conv1D(256, kernel_size=3, padding="causal", activation="relu",
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    shared = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── Price Head (regression) ───────────────────────
    p = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="price_fc1")(shared)
    p = layers.Dropout(dropout_rate)(p)
    p = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="price_fc2")(p)
    price_out = layers.Dense(1, activation="linear", name="price_output")(p)

    # ── Trend Head (binary classification) ───────────
    t = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="trend_fc1")(shared)
    t = layers.Dropout(dropout_rate)(t)
    t = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="trend_fc2")(t)
    trend_out = layers.Dense(1, activation="sigmoid", name="trend_output")(t)

    model = keras.Model(inputs=inp, outputs=[price_out, trend_out], name="StockCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "price_output": "mse",
            "trend_output": "binary_crossentropy",
        },
        loss_weights={
            "price_output": 1.0,
            "trend_output": 2.0,      # weight classification higher
        },
        metrics={
            "price_output": ["mae"],
            "trend_output": ["accuracy"],
        },
    )
    return model


def get_callbacks(checkpoint_path: str) -> list:
    """Standard training callbacks."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_trend_output_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_trend_output_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
