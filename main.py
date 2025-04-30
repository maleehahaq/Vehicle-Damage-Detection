import numpy as np
import tensorflow as tf
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your dataset directories
train_dir = '/content/drive/MyDrive/DIP/project/data3a/training'
validation_dir = '/content/drive/MyDrive/DIP/project/data3a/validation'
test_dir = '/content/drive/MyDrive/DIP/project/data3a/test'

# Image dimensions expected by InceptionV3
img_width, img_height = 299, 299

# Define the number of classes (minor, moderate, severe)
num_classes = 3

# Load the InceptionV3 model without the top (fully connected) layers
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add custom top layers for vehicle damage detection and classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Define the model to train
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the weights of the InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with appropriate loss function and optimizer
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Data augmentation for validation and test sets (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load and preprocess training, validation, and test data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=True  # Ensure shuffling is explicit
    )

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
 # Save the trained model
model.save('detection_model.keras')

