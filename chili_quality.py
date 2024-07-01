import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# Define directories
train_dir = './train'
validation_dir = './validation'

# Parameters for training
batch_size = 32
img_size = (150, 150)
epochs = 20

# Data augmentation and normalization
train_data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using ImageDataGenerator
train_generator = train_data.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Flow validation images in batches using ImageDataGenerator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax')  # 4 output classes (Grade1 to Grade4)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy}')

# Function to load and preprocess image
def load_and_process_image(image_path, img_size=(150, 150)):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

# Path to the image you want to classify
image_path = './test/IMG_0469.JPG'

# Load and preprocess the image
processed_image = load_and_process_image(image_path)

# Make prediction
prediction = model.predict(processed_image)

# Decode prediction
grades = ['Grade1', 'Grade2', 'Grade3', 'Grade4']
predicted_grade = grades[np.argmax(prediction)]

print(f'Predicted chili grade: {predicted_grade}')
