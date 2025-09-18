"""
TensorFlow Basics: a gentle, runnable intro

Theory (read this first):
- TensorFlow represents data as Tensors (multidimensional arrays) and composes operations
    into compute graphs. Keras provides a high-level API for building and training models.
- Use tf.data to build efficient input pipelines (batching, shuffling, prefetching). Train
    with model.fit, then save models in the native .keras format in Keras 3.

This script teaches:
- Importing TensorFlow safely (prints guidance if TF is missing)
- Working with Tensors and operations
- Building and training a simple Keras model
- Using tf.data for input pipelines
- Saving and loading models

Run it directly. If TensorFlow isn't installed, you'll see instructions to install.
"""
from __future__ import annotations
import sys
from textwrap import dedent


def try_import_tf():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except Exception as e:
        print("TensorFlow is not available in this environment.")
        print("Install it with:")
        print("  pip install tensorflow")
        print("Then re-run this script.")
        print("\nDetails:", e)
        return None


def tensors_demo(tf):
    print("\nTensors and basic ops")
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    print("a:", a.numpy())
    print("b:", b.numpy())
    print("a + b:", tf.add(a, b).numpy())
    print("a @ b:", tf.matmul(a, b).numpy())


def make_dataset(tf, n=512):
    import numpy as np
    X = np.random.randn(n, 3).astype("float32")
    w_true = np.array([[2.0], [-1.0], [0.5]], dtype="float32")
    y = X @ w_true + 0.1 * np.random.randn(n, 1).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(1000).batch(32)
    return ds


def build_model(tf):
    from tensorflow.keras import layers
    model = tf.keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="mse", metrics=["mae"])
    return model


def train_and_save(tf):
    print("\nTraining a small Keras model on synthetic data")
    ds = make_dataset(tf)
    model = build_model(tf)
    history = model.fit(ds, epochs=3, verbose=0)
    print("Final metrics:", {k: float(v[-1]) for k, v in history.history.items()})

    out_file = "./out-tf-basics.keras"  # Keras 3 native format requires .keras extension
    model.save(out_file)
    print("Saved model to:", out_file)

    print("\nLoading model back and evaluating")
    loaded = tf.keras.models.load_model(out_file)
    loss, mae = loaded.evaluate(ds, verbose=0)
    print({"loss": float(loss), "mae": float(mae)})


def main(argv):
    print(dedent(
        """
        TensorFlow basics demo
        - We'll import TF, show simple tensor ops, train a tiny Keras model, and save/load it.
        - If TensorFlow isn't installed, you'll see guidance to install it.
        """
    ).strip())

    tf = try_import_tf()
    if tf is None:
        return 0

    tensors_demo(tf)
    train_and_save(tf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

