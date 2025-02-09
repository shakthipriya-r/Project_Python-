import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Adjust these paths and parameters according to your dataset
train_data_dir = r'D:\ss\2023\code\curr\train'
validation_data_dir = r'D:\ss\2023\code\curr\val'
img_width, img_height = 224, 224
batch_size = 32
num_classes = 7  # Adjust based on the number of classes in your dataset

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the modified model
model = Sequential()

# Convolutional layers (DNN part)
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

# Fully connected layers
model.add(Dense(units=800, activation='relu'))  # Replace LSTM with Dense layer

# Additional Dense layers if needed
# model.add(Dense(units=512, activation='relu'))

model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))

# Plot training accuracy
plt.subplot(1, 2, 1)
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

# Plot validation accuracy
plt.subplot(1, 2, 2)
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 4))

# Plot training loss
plt.subplot(1, 2, 1)
if 'loss' in history.history:
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

# Plot validation loss
plt.subplot(1, 2, 2)
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'])
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# Save the model if needed
model.save("modified_model_savedmodel", save_format="tf")
print("Training Ended")
