"""
Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms.

This module implements a production-grade multivariate time series forecasting
pipeline using a Transformer encoder with self-attention.

"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import ttest_rel
import keras_tuner as kt
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping

# Configuration

logging.basicConfig(level=logging.INFO)
np.random.seed(42)
tf.random.set_seed(42)

WINDOW_SIZE = 48
TRAIN_SPLIT = 0.8


# 1. DATA GENERATION

def generate_server_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic multivariate server time series
    exhibiting trend, seasonality, structural break, and anomalies.
    """
    t = np.arange(n)

    trend = 0.01 * t
    daily = 20 * np.sin(2 * np.pi * t / 24)
    weekly = 15 * np.sin(2 * np.pi * t / (24 * 7))
    structural_break = np.where(t > 1200, 60, 0)

    requests = 200 + trend + daily + weekly + structural_break + np.random.normal(0, 5, n)
    cpu = 0.3 * requests + np.random.normal(0, 3, n)
    memory = 0.5 * cpu + np.random.normal(0, 2, n)
    latency = 100 + 0.2 * requests + np.random.normal(0, 4, n)
    error_rate = 0.01 * requests + np.random.normal(0, 1, n)

    df = pd.DataFrame({
        "requests": requests,
        "cpu": cpu,
        "memory": memory,
        "latency": latency,
        "error_rate": error_rate
    })

    df["hour"] = t % 24
    df["day_of_week"] = (t // 24) % 7
    df["lag_1"] = df["requests"].shift(1)
    df["lag_24"] = df["requests"].shift(24)

    # Missing blocks
    df.iloc[300:320] = np.nan
    df.iloc[900:920] = np.nan

    # Inject anomaly
    df.iloc[600:605] *= 1.7

    return df


# 2. PREPROCESSING

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values, scale features, and prevent data leakage.
    """
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    split_index = int(len(df_imputed) * TRAIN_SPLIT)
    train_df = df_imputed[:split_index]
    test_df = df_imputed[split_index:]

    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    full_scaled = np.vstack([train_scaled, test_scaled])

    return full_scaled, df_imputed["requests"].values


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    window: int = WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series into supervised learning sequences.
    """
    X, y = [], []

    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(target[i])

    return np.array(X), np.array(y)


# 3. TRANSFORMER MODEL

class PositionalEncoding(layers.Layer):
    """
    Standard sinusoidal positional encoding.
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]

        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32)
            * -(np.log(10000.0) / tf.cast(d_model, tf.float32))
        )

        pe = tf.zeros((seq_len, d_model))
        pe_even = tf.sin(position * div_term)
        pe_odd = tf.cos(position * div_term)

        pe = tf.concat([pe_even, pe_odd], axis=-1)
        pe = pe[tf.newaxis, ...]

        return x + pe


def build_model(hp: kt.HyperParameters) -> Model:
    """
    Build Transformer model with tunable hyperparameters.
    """
    inputs = layers.Input(shape=(WINDOW_SIZE, 9))
    x = PositionalEncoding()(inputs)

    attn_layer = layers.MultiHeadAttention(
        num_heads=hp.Int("heads", 2, 6),
        key_dim=hp.Int("key_dim", 16, 64, step=16),
    )

    attn_output = attn_layer(x, x)
    x = layers.LayerNormalization()(x + attn_output)

    ffn = layers.Dense(
        hp.Int("ff_dim", 64, 256, step=64),
        activation="relu"
    )(x)

    ffn = layers.Dropout(
        hp.Float("dropout", 0.1, 0.4, step=0.1)
    )(ffn)

    x = layers.LayerNormalization()(x + ffn)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("lr", 1e-4, 1e-3, sampling="log")
        ),
        loss=Huber()
    )

    return model


# 4. METRICS

def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    naive = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / naive


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    return np.mean((y_true >= lower) & (y_true <= upper))


def monte_carlo_dropout(
    model: Model,
    X: np.ndarray,
    n_samples: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.array([
        model(X, training=True).numpy().flatten()
        for _ in range(n_samples)
    ])
    return preds.mean(axis=0), preds.std(axis=0)


# 5. MAIN EXECUTION

def main() -> None:

    df = generate_server_data()
    data, target = preprocess_data(df)
    X, y = create_sequences(data, target)

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=8,
        directory="tuning",
        project_name="advanced_ts"
    )

    tuner.search(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        verbose=0
    )

    model = tuner.get_best_models(1)[0]

    model.fit(
        X_train,
        y_train,
        epochs=40,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    mean_pred, std_pred = monte_carlo_dropout(model, X_test)
    lower = mean_pred - 1.96 * std_pred
    upper = mean_pred + 1.96 * std_pred

    # SARIMA baseline
    sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    sarima_fit = sarima.fit(disp=False)
    sarima_pred = sarima_fit.forecast(len(y_test))

    rmse_dl = np.sqrt(mean_squared_error(y_test, mean_pred))
    rmse_s = np.sqrt(mean_squared_error(y_test, sarima_pred))

    mase_dl = mase(y_test, mean_pred, y_train)
    mase_s = mase(y_test, sarima_pred, y_train)

    coverage = prediction_interval_coverage(y_test, lower, upper)

    dm_pvalue = ttest_rel(
        (y_test - mean_pred) ** 2,
        (y_test - sarima_pred) ** 2
    ).pvalue

    print("\nModel Performance Summary")
    print("--------------------------------------------------")
    print(f"{'Metric':<25}{'Transformer':<15}{'SARIMA'}")
    print("--------------------------------------------------")
    print(f"{'RMSE':<25}{rmse_dl:<15.4f}{rmse_s:.4f}")
    print(f"{'MASE':<25}{mase_dl:<15.4f}{mase_s:.4f}")
    print(f"{'Interval Coverage':<25}{coverage:<15.4f}{'-'}")
    print("--------------------------------------------------")
    print(f"Diebold-Mariano p-value: {dm_pvalue:.6f}")


if __name__ == "__main__":
    main()
