import matplotlib
matplotlib.use('Agg')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adagrad
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from cancernet.cancernet import CancerNet
from cancernet import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_EPOCHS = 40
INIT_LR = 1e-2
BS = 32

# Load dataset paths
train_paths = list(paths.list_images(config.TRAIN_PATH))
val_paths = list(paths.list_images(config.VAL_PATH))
test_paths = list(paths.list_images(config.TEST_PATH))

# Check if datasets are loaded
if not train_paths or not val_paths or not test_paths:
    raise ValueError('One or more dataset paths are empty. Please check your dataset directories.')

# Prepare labels
train_labels = [int(p.split(os.path.sep)[-2]) for p in train_paths]
if len(train_labels) == 0:
    raise ValueError('No training labels found. Please check your dataset.')

train_labels = to_categorical(train_labels)

class_totals = train_labels.sum(axis=0)
class_weight = class_totals.max() / class_totals

# Data augmentation
train_aug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_aug = ImageDataGenerator(rescale=1 / 255.0)

# Create data generators
train_gen = train_aug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode='categorical',
    target_size=(48, 48),
    color_mode='rgb',
    shuffle=True,
    batch_size=BS
)

val_gen = val_aug.flow_from_directory(
    config.VAL_PATH,
    class_mode='categorical',
    target_size=(48, 48),
    color_mode='rgb',
    shuffle=False,
    batch_size=BS
)

test_gen = val_aug.flow_from_directory(
    config.TEST_PATH,
    class_mode='categorical',
    target_size=(48, 48),
    color_mode='rgb',
    shuffle=False,
    batch_size=BS
)

# Build model
model = CancerNet.build(width=48, height=48, depth=3, classes=2)
opt = Adagrad(learning_rate=INIT_LR)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train model
M = model.fit(
    train_gen,
    steps_per_epoch=len(train_paths) // BS,
    validation_data=val_gen,
    validation_steps=len(val_paths) // BS,
    class_weight=class_weight,
    epochs=NUM_EPOCHS
)

# Evaluate the model
print('Now evaluating the model')
test_gen.reset()
pred_indices = model.predict(test_gen, steps=(len(test_paths) // BS) + 1)
pred_indices = np.argmax(pred_indices, axis=1)

print(classification_report(test_gen.classes, pred_indices, target_names=test_gen.class_indices.keys()))

cm = confusion_matrix(test_gen.classes, pred_indices)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print(cm)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')

# Plotting training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, NUM_EPOCHS), M.history['loss'], label='train_loss')
plt.plot(np.arange(0, NUM_EPOCHS), M.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, NUM_EPOCHS), M.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, NUM_EPOCHS), M.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('plot.png')
