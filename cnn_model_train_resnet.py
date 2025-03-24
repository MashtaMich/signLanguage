import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
import matplotlib.pyplot as plt
import kerastuner as kt

RAW_DATA_DIR = 'dataset'
BASE_SPLIT_DIR = 'splitted_dataset'
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
MAX_EPOCHS = 10

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
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

# Visualize some augmented images
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
    # 1) Load a ResNet base model (without the top fully connected layers)
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    # 2) Optionally freeze some portion of ResNet layers
    #    For example, let's tune how many layers to keep trainable.
    #    You can also freeze *all* layers and only train the new head for a while.
    #    The total number of layers in ResNet50 is around 175. 
    #    We'll pick a tuneable "unfreeze_from" layer index in [100 .. 165] for demonstration.
    unfreeze_from = hp.Int('unfreeze_from', min_value=100, max_value=165, step=15)

    for i, layer in enumerate(base_model.layers):
        if i < unfreeze_from:
            layer.trainable = False
        else:
            layer.trainable = True

    # 3) Create the new classification head
    x = tf.keras.layers.Flatten()(base_model.output)

    # Tune the number of units in the FC layer
    dense_units = hp.Int('dense_units', 64, 512, step=64)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)

    # Tune dropout
    dropout_rate = hp.Float('dropout_rate', 0.2, 0.6, step=0.1)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Final output layer for your classes
    # Make sure num_classes matches your dataset
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # 4) Build the model
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # 5) Compile the model (tune the learning rate if desired)
    learning_rate = hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# --- Step 4: Set up KerasTuner ---
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=11,
    executions_per_trial=1,
    directory='kt_tuner_dir',
    project_name='sign_language_tuning_resnet'
)


# Early stopping callback
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(train_generator, epochs=MAX_EPOCHS, validation_data=validation_generator, callbacks=[early_stop])
tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=MAX_EPOCHS,
    callbacks=[early_stop]
)

test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

best_model.save("resnet_sign_language_best_model.h5")
print("Model saved as resnet_sign_language_best_model.h5")