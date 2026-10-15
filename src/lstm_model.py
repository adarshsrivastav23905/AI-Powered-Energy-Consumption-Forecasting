from __future__ import annotations

"""Optional advanced LSTM forecasting module.

This file is kept as a future upgrade path for students who want a deep-learning
version of the project. It is intentionally lightweight and only imports TensorFlow
when the advanced model is actually executed.
"""


def build_lstm_model(input_shape, units=64):
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for the LSTM version. Install it with `pip install tensorflow`."
        ) from exc

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units // 2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model
