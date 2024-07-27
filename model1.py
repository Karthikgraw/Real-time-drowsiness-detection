import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24, 24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

BS = 32
TS = (24, 24)

train_batch = generator('D:/Drowsiness detection/dataset_new/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('D:/Drowsiness detection/dataset_new/test', shuffle=True, batch_size=BS, target_size=TS)

SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS

print(SPE, VS)

# Plotting the distribution of the dataset
def plot_data_distribution(generator, title):
    labels_count = generator.classes
    label_names = list(generator.class_indices.keys())
    sns.countplot(x=labels_count)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(ticks=range(len(label_names)), labels=label_names)
    plt.show()

plot_data_distribution(train_batch, 'Training Data Distribution')
plot_data_distribution(valid_batch, 'Validation Data Distribution')

# Plotting a heatmap of a few images from the training data
def plot_sample_images(generator, title, nrows=3, ncols=3):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    axes = axes.flatten()
    for img, label in generator:
        for i, ax in enumerate(axes):
            ax.imshow(img[0].reshape(24, 24), cmap='gray')
            ax.set_title(f'Label: {list(generator.class_indices.keys())[np.argmax(label[0])]}')
            ax.axis('off')
        break
    plt.suptitle(title)
    plt.show()

plot_sample_images(train_batch, 'Sample Images from Training Data')

# Plotting only "Closed" eye images from the training data
def plot_closed_eye_images(generator, title, nrows=3, ncols=3):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    axes = axes.flatten()
    count = 0
    for img, label in generator:
        if np.argmax(label[0]) == 0:  # Assuming 0 is Closed
            axes[count].imshow(img[0].reshape(24, 24), cmap='gray')
            axes[count].set_title('Closed')
            axes[count].axis('off')
            count += 1
            if count == nrows * ncols:
                break
    plt.suptitle(title)
    plt.show()

plot_closed_eye_images(train_batch, 'Sample Images of Closed Eyes from Training Data')

# Model definition
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

#model.save('models/cnnCat2.h5', overwrite=True)

# Evaluation
Y_pred = model.predict(valid_batch, len(valid_batch))
y_pred = np.argmax(Y_pred, axis=1)
y_true = valid_batch.classes

print('Classification Report')
print(classification_report(y_true, y_pred, target_names=valid_batch.class_indices.keys()))

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
