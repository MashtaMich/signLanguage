import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
import matplotlib.pyplot as plt
import kerastuner as kt  # Make sure to install keras-tuner (pip install keras-tuner)

# --- Parameters ---
RAW_DATA_DIR = 'dataset'     # Raw images folder: contains subdirectories per class
BASE_SPLIT_DIR = 'splitted_dataset'      # Base directory for train, validation, test splits
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
MAX_EPOCHS = 20

# --- Step 1: Split raw data into train, val, and test directories ---
def split_dataset(raw_dir, base_dir, train_ratio, val_ratio, test_ratio):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
    for split in [train_dir, val_dir, test_dir]:
        os.makedirs(split, exist_ok=True)
    
    for class_name in os.listdir(raw_dir):
        class_raw_dir = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_raw_dir):
            continue

        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        file_names = os.listdir(class_raw_dir)
        random.shuffle(file_names)
        total = len(file_names)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        for i, fname in enumerate(file_names):
            src_path = os.path.join(class_raw_dir, fname)
            if i < train_end:
                dst_path = os.path.join(train_dir, class_name, fname)
            elif i < val_end:
                dst_path = os.path.join(val_dir, class_name, fname)
            else:
                dst_path = os.path.join(test_dir, class_name, fname)
            shutil.copy(src_path, dst_path)
    
    print(f"Dataset split completed. Train, Val, and Test folders created in '{base_dir}'.")
    return train_dir, val_dir, test_dir

if not os.path.exists(BASE_SPLIT_DIR):
    train_dir, val_dir, test_dir = split_dataset(RAW_DATA_DIR, BASE_SPLIT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
else:
    train_dir = os.path.join(BASE_SPLIT_DIR, 'train')
    val_dir = os.path.join(BASE_SPLIT_DIR, 'val')
    test_dir = os.path.join(BASE_SPLIT_DIR, 'test')
    print(f"Using existing split dataset in '{BASE_SPLIT_DIR}'.")

# --- Step 2: Create ImageDataGenerators with augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Optional: Visualize some augmented images
images, _ = next(train_generator)
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.suptitle("Example Augmented Images")
plt.show()

# --- Step 3: Define the model-building function for KerasTuner ---
num_classes = len(train_generator.class_indices)
print("Detected classes:", train_generator.class_indices)

def build_model(hp):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    
    # Tune the number of convolutional layers (2 to 4)
    for i in range(hp.Int('conv_layers', 2, 4)):
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32)
        model.add(layers.Conv2D(filters, (3,3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 1e-4, 1e-2, sampling='log'))))
        model.add(layers.MaxPooling2D((2,2)))
        dropout_rate = hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Flatten())
    dense_units = hp.Int('dense_units', 64, 256, step=64)
    model.add(layers.Dense(dense_units, activation='relu',
                           kernel_regularizer=regularizers.l2(hp.Float('l2_reg_dense', 1e-4, 1e-2, sampling='log'))))
    dense_dropout = hp.Float('dense_dropout', 0.3, 0.7, step=0.1)
    model.add(layers.Dropout(dense_dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Step 4: Set up KerasTuner ---
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='kt_tuner_dir',
    project_name='sign_language_tuning'
)

# Early stopping callback
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Perform hyperparameter search
tuner.search(train_generator, epochs=MAX_EPOCHS, validation_data=validation_generator, callbacks=[early_stop])

# Print a summary of the search results
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# --- Step 5: Fine-tune the Best Model and Evaluate ---
history = best_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=MAX_EPOCHS,
    callbacks=[early_stop]
)

test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the final best model
best_model.save("cnn_sign_language_best_model.h5")
print("Model saved as cnn_sign_language_best_model.h5")