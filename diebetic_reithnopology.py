import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import perf_counter
from pathlib import Path
from IPython.display import Image, display, Markdown

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf

import seaborn as sns

def printmd(string):
    display(Markdown(string))

imageDir = Path('New folder/gaussian_filtered_images/gaussian_filtered_images')
image_path: str = ""
def communiator_1(imae_path):
    global image_path
    image_path = imae_path


filepaths = list(imageDir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

image_df = pd.concat([filepaths, labels], axis=1)
image_df = image_df.sample(frac=1).reset_index(drop = True)



fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[i]))
    ax.set_title(image_df.Label[i])

plt.tight_layout()



vc = image_df['Label'].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(x=vc.index, y=vc, palette="rocket")
plt.title("No. of pictures in each category", fontsize=15)

plt.show()
trainImages = None
valImages = None
testImages = None
your_test_images = None
your_image_df = pd.DataFrame({"Filepath": [image_path], "Label": ["unclassified"]})
def createGen():
    global trainImages, valImages, testImages
    trainGen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

    testGen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    trainImages = trainGen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    valImages = trainGen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    testImages = testGen.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    your_test_images = testGen.flow_from_dataframe(
        dataframe=your_image_df,
        x_col='Filepath',
        y_col= 'Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=1,  # Set batch_size to 1 for a single image
        shuffle=False
    )

    return trainGen, testGen, trainImages, valImages, testImages, your_test_images

def getModel(model):
    kwargs = {
        'input_shape':(224, 224, 3),
        'include_top':False,
        'weights':'imagenet',
        'pooling':'avg'
    }

    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False

    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
train_df, test_df = train_test_split(image_df, train_size=0.9, shuffle=True, random_state=1)

createGen()
# pretrained_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling='avg'
# )
# pretrained_model.trainable = False
#
# inputs = pretrained_model.input
#
# x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
#
# outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
#
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# history = model.fit(
#     trainImages,
#     validation_data=valImages,
#     batch_size = 32,
#     epochs=10,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=2,
#             restore_best_weights=True
#         )
#     ]
# )
model = tf.keras.models.load_model("model.h5")
# pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
# plt.title("Accuracy VS val_accuracy")
# plt.show()

# pd.DataFrame(history.history)[['loss','val_loss']].plot()
# plt.title("Loss VS val_loss")
# plt.show()

results = model.evaluate(testImages, verbose=0)

print('> ## Test Loss |> {:.5f}'.format(results[0]))
print('> ## Accuracy |> {:.2f}%'.format(results[1] *100))

pred = model.predict(testImages)
pred = np.argmax(pred,axis=1)

labels = (trainImages.class_indices)
labels = dict((v, k) for k, v in labels.items())

pred = [labels[k] for k in pred]

print(f'The first 10 predictions:\n{pred[:10]}')

y_test = list(test_df.Label)
print(classification_report(y_test, pred))
#
cf_matrix = confusion_matrix(y_test, pred, normalize='true')
plt.figure(figsize = (15, 10))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)))
plt.title('Normalized Confusion Matrix')

plt.show()

import matplotlib.pyplot as plt

# Access the image path and prediction from your DataFrame
img_path = your_image_df['Filepath'].iloc[0]  # Assuming there's only one image in your DataFrame
prediction = pred[0]  # Assuming pred contains predictions for the corresponding image

# Display the image with its prediction




