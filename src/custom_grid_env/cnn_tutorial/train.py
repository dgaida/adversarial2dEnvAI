import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def load_data(data_dir="data"):
    """Loads images and labels from the data directory."""
    images = []
    labels = []
    class_names = ["dog", "flower", "background"]

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(
                f"Warning: Directory {class_dir} not found. Skipping class {class_name}."
            )
            continue
        for img_name in os.listdir(class_dir):
            if img_name.endswith(".png"):
                img_path = os.path.join(class_dir, img_name)
                img = keras.utils.load_img(img_path, target_size=(64, 64))
                img_array = keras.utils.img_to_array(img)
                images.append(img_array)
                labels.append(label)

    images = np.array(images) / 255.0  # Normalize to [0, 1]
    labels = np.array(labels)
    return images, labels, class_names


def train_model(epochs=10, batch_size=32):
    """Trains a simple CNN and saves performance visualizations."""
    print("Loading data...")
    X, y, class_names = load_data()

    if len(X) == 0:
        print("Error: No data loaded. Check data generation.")
        return None

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simple CNN Architecture
    model = keras.Sequential(
        [
            layers.Input(shape=(64, 64, 3)),
            # Small CNN to keep it simple and allow for some errors
            layers.Conv2D(8, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(3, activation="softmax"),  # 3 classes: dog, flower, background
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    print("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )

    # --- Visualization ---
    os.makedirs("results", exist_ok=True)

    # 1. Plot Accuracy and Loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_metrics.png")
    print("Saved training metrics to results/training_metrics.png")

    # 2. Confusion Matrix
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_val, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    print("Saved confusion matrix to results/confusion_matrix.png")

    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "model.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    return model


if __name__ == "__main__":
    # Ensure data exists
    if not os.path.exists("data"):
        print("Error: 'data' directory not found. Run data_generation.py first.")
    else:
        train_model(epochs=15)
