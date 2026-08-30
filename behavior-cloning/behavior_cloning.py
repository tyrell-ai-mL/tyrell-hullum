import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def load_data(csv_path):
    columns = [
        "center", "left", "right", "steering",
        "throttle", "brake", "speed"
    ]

    data = pd.read_csv(csv_path, names=columns)

    images = []
    steering_angles = []

    for _, row in data.iterrows():
        image_path = Path(row["center"].strip())
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        steering_angles.append(float(row["steering"]))

    return np.array(images), np.array(steering_angles)


def build_model(input_shape):
    model = Sequential([
        Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape),
        Conv2D(24, (5, 5), strides=(2, 2), activation="relu"),
        Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
        Conv2D(48, (5, 5), strides=(2, 2), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        Flatten(),
        Dense(100, activation="relu"),
        Dropout(0.5),
        Dense(50, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1)
    ])

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a behavior cloning model for autonomous steering."
    )
    parser.add_argument("--csv", default="driving_log.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", default="behavior_cloning_model.keras")
    args = parser.parse_args()

    images, steering = load_data(args.csv)

    if len(images) == 0:
        raise RuntimeError(
            "No training images were found. Check the paths in driving_log.csv."
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        images,
        steering,
        test_size=0.2,
        random_state=42
    )

    model = build_model(X_train.shape[1:])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="mse"
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=args.epochs,
        batch_size=32,
        shuffle=True
    )

    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()