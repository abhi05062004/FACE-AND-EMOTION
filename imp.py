import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import matplotlib.pyplot as plt

def train_emotion_detection_model():
    """
    Trains a Convolutional Neural Network (CNN) for emotion detection.

    This function assumes your dataset is organized into subfolders,
    where each subfolder represents an emotion label and contains
    images corresponding to that emotion.

    Example dataset structure:
    your_dataset_folder/
    ├── angry/
    │   ├── img_001.jpg
    │   └── ...
    ├── happy/
    │   ├── img_001.jpg
    │   └── ...
    └── ... (other emotion folders)
    """

    # --- Configuration ---
    # Path to your dataset's root directory
    # IMPORTANT: Replace this with the actual path to your dataset!
    dataset_path = 'C:\Users\ABHIJITH\Documents\face detection\test' # e.g., 'C:/Users/ABHIJITH/Documents/MyEmotionDataset'

    # Image dimensions for model input
    img_height, img_width = 48, 48 # Common size for emotion datasets like FER-2013
    batch_size = 32
    epochs = 15 # Number of training epochs. You might need more for better accuracy.

    # Define the emotion labels (these should match your dataset folder names)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    num_classes = len(emotion_labels)

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        print("Please update 'dataset_path' variable to point to your actual dataset directory.")
        return

    print(f"[INFO] Using dataset from: {dataset_path}")
    print(f"[INFO] Expected emotion labels: {emotion_labels}")

    # --- Data Preprocessing and Augmentation ---
    # ImageDataGenerator is used to load images from directories and apply augmentations
    # for better model generalization.
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values to [0, 1]
        rotation_range=10,       # Randomly rotate images by up to 10 degrees
        width_shift_range=0.1,   # Randomly shift image horizontally
        height_shift_range=0.1,  # Randomly shift image vertically
        shear_range=0.1,         # Apply shear transformation
        zoom_range=0.1,          # Randomly zoom in/out
        horizontal_flip=True,    # Randomly flip images horizontally
        fill_mode='nearest',     # Strategy for filling in new pixels created by transformations
        validation_split=0.2     # Use 20% of data for validation
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale', # Emotion models often use grayscale images
        class_mode='categorical', # For multi-class classification
        subset='training',        # Specify this is the training set
        classes=emotion_labels,   # Explicitly set class order
        seed=42
    )

    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',      # Specify this is the validation set
        classes=emotion_labels,   # Explicitly set class order
        seed=42
    )

    # Verify that the class indices match the labels
    print(f"[INFO] Training class indices: {train_generator.class_indices}")
    print(f"[INFO] Validation class indices: {validation_generator.class_indices}")

    # --- Model Definition (Simple CNN Architecture) ---
    print("[INFO] Building CNN model...")
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25), # Dropout for regularization

        # Second Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Third Conv Block
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Flatten and Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Output layer with softmax for probabilities
    ])

    # --- Compile the Model ---
    # Optimizer: Adam is a good general-purpose optimizer
    # Loss: Categorical crossentropy for multi-class classification
    # Metrics: Accuracy to monitor performance
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary() # Print model summary to console

    # --- Train the Model ---
    print("[INFO] Starting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    print("[INFO] Model training finished.")

    # --- Save the Trained Model ---
    model_save_path = 'emotion_model.h5'
    model.save(model_save_path)
    print(f"[INFO] Trained model saved to: {model_save_path}")

    # --- Plot Training History (Optional) ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print("[INFO] Training complete. You can now use 'emotion_model.h5' for inference.")

if __name__ == "__main__":
    train_emotion_detection_model()
