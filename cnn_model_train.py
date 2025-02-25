# import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import keras_tuner as kt

# # Load the pickled datasets
# with open("train.pickle", "rb") as f:
#     train_images, train_labels = pickle.load(f)
# with open("val.pickle", "rb") as f:
#     val_images, val_labels = pickle.load(f)
# with open("test.pickle", "rb") as f:
#     test_images, test_labels = pickle.load(f)

# # Convert to NumPy arrays and reshape images to add channel dimension
# train_images = np.array(train_images).reshape(-1, 50, 50, 1)
# val_images = np.array(val_images).reshape(-1, 50, 50, 1)
# test_images = np.array(test_images).reshape(-1, 50, 50, 1)

# train_labels = np.array(train_labels)
# val_labels = np.array(val_labels)
# test_labels = np.array(test_labels)

# # Determine the number of classes from the training labels
# num_classes = len(np.unique(train_labels))

# def build_model(hp):
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(50, 50, 1)))
    
#     # First convolutional layer: number of filters is tunable
#     hp_filters1 = hp.Int('filters1', min_value=16, max_value=64, step=16)
#     model.add(layers.Conv2D(filters=hp_filters1, kernel_size=(3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
#     # Second convolutional layer
#     hp_filters2 = hp.Int('filters2', min_value=32, max_value=128, step=32)
#     model.add(layers.Conv2D(filters=hp_filters2, kernel_size=(3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
#     model.add(layers.Flatten())
    
#     # Dense layer with tunable number of units
#     hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
#     model.add(layers.Dense(units=hp_units, activation='relu'))
    
#     # Tunable dropout rate
#     hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
#     model.add(layers.Dropout(rate=hp_dropout))
    
#     # Output layer
#     model.add(layers.Dense(num_classes, activation='softmax'))
    
#     # Tunable learning rate for Adam optimizer
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# # Set up the KerasTuner search. Here we use Hyperband.
# tuner = kt.Hyperband(build_model,
#                      objective='val_accuracy',
#                      max_epochs=20,
#                      factor=3,
#                      directory='kt_dir',
#                      project_name='sign_language')

# # Callback to stop training early if no improvement is seen
# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# # Run the hyperparameter search
# tuner.search(train_images, train_labels,
#              epochs=20,
#              validation_data=(val_images, val_labels),
#              callbacks=[stop_early])

# # Get the best hyperparameters and display them
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# print("The hyperparameter search is complete.")
# print(f"Best filters1: {best_hps.get('filters1')}")
# print(f"Best filters2: {best_hps.get('filters2')}")
# print(f"Best units: {best_hps.get('units')}")
# print(f"Best dropout: {best_hps.get('dropout')}")
# print(f"Best learning rate: {best_hps.get('learning_rate')}")

# # Build the best model and train it further if desired
# model = tuner.hypermodel.build(best_hps)
# history = model.fit(train_images, train_labels,
#                     epochs=20,
#                     validation_data=(val_images, val_labels))

# # Evaluate the final model on the test set
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Test accuracy:", test_acc)

# model.save("sign_language_model.h5")
# print("Model saved as sign_language_model.h5")
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# Load the pickled datasets
with open("train2.pickle", "rb") as f:
    train_images, train_labels = pickle.load(f)
with open("val2.pickle", "rb") as f:
    val_images, val_labels = pickle.load(f)
with open("test2.pickle", "rb") as f:
    test_images, test_labels = pickle.load(f)

# Convert to NumPy arrays and reshape images to (50, 50, 1)
train_images = np.array(train_images).reshape(-1, 50, 50, 1)
val_images = np.array(val_images).reshape(-1, 50, 50, 1)
test_images = np.array(test_images).reshape(-1, 50, 50, 1)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# Determine the number of classes
num_classes = len(np.unique(train_labels))

def build_model(hp):
    model = keras.Sequential()
    
    # Input layer with normalization (scale pixel values to [0, 1])
    model.add(layers.Input(shape=(50, 50, 1)))
    model.add(layers.Rescaling(1./255))
    
    # Tune the L2 regularization factor (weight decay)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='LOG', default=1e-4)
    
    # Simplified convolutional block with L2 regularization
    hp_filters = hp.Int('filters', min_value=16, max_value=32, step=16)
    model.add(layers.Conv2D(filters=hp_filters, kernel_size=(3, 3), activation='relu', 
                            padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Optionally include a second convolutional block
    if hp.Boolean('second_conv'):
        model.add(layers.Conv2D(filters=hp_filters * 2, kernel_size=(3, 3), activation='relu', 
                                padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Flatten())
    
    # Dense layer with L2 regularization
    hp_units = hp.Int('units', min_value=16, max_value=64, step=16)
    model.add(layers.Dense(units=hp_units, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(l2_reg)))
    
    # Dropout layer for regularization
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.3, step=0.1)
    model.add(layers.Dropout(rate=hp_dropout))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Tune learning rate for Adam optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Set up KerasTuner using Hyperband
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='kt_dir',
    project_name='sign_language_regularized'
)

# Callbacks: early stopping and reduce learning rate on plateau
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Run hyperparameter search
tuner.search(train_images, train_labels,
             epochs=20,
             validation_data=(val_images, val_labels),
             callbacks=[early_stopping, reduce_lr])

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Hyperparameter search complete.")
print(f"Best filters: {best_hps.get('filters')}")
print(f"Second conv included: {best_hps.get('second_conv')}")
print(f"Best units: {best_hps.get('units')}")
print(f"Best dropout: {best_hps.get('dropout')}")
print(f"Best L2 regularizer: {best_hps.get('l2_reg')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Build and train the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_images, train_labels,
                    epochs=20,
                    validation_data=(val_images, val_labels),
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# Save the trained model to a file
model.save("sign_language_model_regularized-2.h5")