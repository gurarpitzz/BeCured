import openai
from flask import Flask, render_template, request, request, redirect, url_for
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import hj
app = Flask(__name__)
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
import google.generativeai as genai


def printmd(string):
    display(Markdown(string))

imageDir = Path('gaussian_filtered_images/gaussian_filtered_images')
image_path = ""
prediction = ""
import tensorflow as tf

import seaborn as sns
from hj import *
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
filepaths = list(imageDir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

image_df = pd.concat([filepaths, labels], axis=1)
image_df = image_df.sample(frac=1).reset_index(drop=True)

vc = image_df['Label'].value_counts()

sns.barplot(x=vc.index, y=vc, palette="rocket")

trainImages = None
valImages = None
testImages = None
your_test_images = None
your_image_df = pd.DataFrame({"Filepath": [image_path], "Label": ["unclassified"]})
train_df, test_df = train_test_split(image_df, train_size=0.9, shuffle=True, random_state=1)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    id = request.form['id']
    age = request.form['age']
    bp = request.form['bp']
    sg = request.form['sg']
    al = request.form['al']
    su = request.form['su']
    rbc = request.form['rbc']
    pc = request.form['pc']
    pcc = request.form['pcc']
    ba = request.form['ba']
    bgr = request.form['bgr']
    bu = request.form['bu']
    sc = request.form['sc']
    sod = request.form['sod']
    pot = request.form['pot']
    hemo = request.form['hemo']
    pcv = request.form['pcv']
    wc = request.form['wc']
    rc = request.form['rc']
    htn = request.form['htn']
    dm = request.form['dm']
    cad = request.form['cad']
    appet = request.form['appet']
    pe = request.form['pe']
    ane = request.form['ane']

    # Save uploaded image
    image_file = request.files['image']
    img_path = None
    if image_file:
        filename = image_file.filename
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(img_path)

    # Construct the message
    messages = f"ID: {id}\nAge: {age}\nBlood Pressure: {bp}\nSpecific Gravity: {sg}\nAlbumin: {al}\nSugar: {su}\nRed Blood Cells: {rbc}\nPus Cell: {pc}\nPus Cell Clumps: {pcc}\nBacteria: {ba}\nBlood Glucose Random: {bgr}\nBlood Urea: {bu}\nSerum Creatinine: {sc}\nSodium: {sod}\nPotassium: {pot}\nHaemoglobin: {hemo}\nPacked Cell Volume: {pcv}\nWhite Blood Cell Count: {wc}\nRed Blood Cell Count: {rc}\nHypertension: {htn}\nDiabetes Mellitus: {dm}\nCoronary Artery Disease: {cad}\nAppetite: {appet}\nPedal Edema: {pe}\nAnemia: {ane}\nImage Path: {image_path}"
    img_path = image_path
    # Do something with the message, like save it to a database or file
    # For now, let's just print it
    print(messages)
    text_1 = hj.kidney_risk_assesment(messages)



    models = {
        "DenseNet121": {"model": tf.keras.applications.DenseNet121, "perf": 0},
        "MobileNetV2": {"model": tf.keras.applications.MobileNetV2, "perf": 0},
        "DenseNet169": {"model": tf.keras.applications.DenseNet169, "perf": 0},
        "DenseNet201": {"model": tf.keras.applications.DenseNet201, "perf": 0},
        "EfficientNetB0": {"model": tf.keras.applications.EfficientNetB0, "perf": 0},
        "EfficientNetB1": {"model": tf.keras.applications.EfficientNetB1, "perf": 0},
        "EfficientNetB2": {"model": tf.keras.applications.EfficientNetB2, "perf": 0},
        "EfficientNetB3": {"model": tf.keras.applications.EfficientNetB3, "perf": 0},
        "EfficientNetB4": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "EfficientNetB5": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "EfficientNetB6": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "EfficientNetB7": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "InceptionResNetV2": {"model": tf.keras.applications.InceptionResNetV2, "perf": 0},
        "InceptionV3": {"model": tf.keras.applications.InceptionV3, "perf": 0},
        "MobileNet": {"model": tf.keras.applications.MobileNet, "perf": 0},
        "MobileNetV2": {"model": tf.keras.applications.MobileNetV2, "perf": 0},
        "MobileNetV3Large": {"model": tf.keras.applications.MobileNetV3Large, "perf": 0},
        "MobileNetV3Small": {"model": tf.keras.applications.MobileNetV3Small, "perf": 0},
        "NASNetMobile": {"model": tf.keras.applications.NASNetMobile, "perf": 0},
        "ResNet101": {"model": tf.keras.applications.ResNet101, "perf": 0},
        "ResNet101V2": {"model": tf.keras.applications.ResNet101V2, "perf": 0},
        "ResNet152": {"model": tf.keras.applications.ResNet152, "perf": 0},
        "ResNet152V2": {"model": tf.keras.applications.ResNet152V2, "perf": 0},
        "ResNet50": {"model": tf.keras.applications.ResNet50, "perf": 0},
        "ResNet50V2": {"model": tf.keras.applications.ResNet50V2, "perf": 0},
        "VGG16": {"model": tf.keras.applications.VGG16, "perf": 0},
        "VGG19": {"model": tf.keras.applications.VGG19, "perf": 0},
        "Xception": {"model": tf.keras.applications.Xception, "perf": 0}
    }
    createGen()
    model = tf.keras.models.load_model("model.h5")
    # pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
    # plt.title("Accuracy VS val_accuracy")
    # plt.show()

    # pd.DataFrame(history.history)[['loss','val_loss']].plot()
    # plt.title("Loss VS val_loss")
    # plt.show()

    results = model.evaluate(testImages, verbose=0)

    print('> ## Test Loss |> {:.5f}'.format(results[0]))
    print('> ## Accuracy |> {:.2f}%'.format(results[1] * 100))

    pred = model.predict(testImages)
    pred = np.argmax(pred, axis=1)

    labels = (trainImages.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    pred = [labels[k] for k in pred]

    print(f'The first 10 predictions:\n{pred[:10]}')

    y_test = list(test_df.Label)
    print(classification_report(y_test, pred))
    #
    cf_matrix = confusion_matrix(y_test, pred, normalize='true')

    import matplotlib.pyplot as plt

    # Access the image path and prediction from your DataFrame
    img_path = your_image_df['Filepath'].iloc[0]  # Assuming there's only one image in your DataFrame
    prediction = pred[0]  # Assuming pred contains predictions for the corresponding image

    # Display the image with its prediction

    #

    pdf_maker(result=prediction, text_1=text_1)

    return 'Form submitted successfully'

def pdf_maker(result, text_1):


    text_2 = "You give detailed risk assesment based on pateint data, you also give precautions, current state, and what to do now"

    message = f"give detailed risk assesment based on pateint data, you also give precautions, current state, and what to do now like you are  medical profectional current retinopology result {result}"




    # library = input("Which Library do ou want to understand")
    genai.configure(api_key="AIzaSyArvzUuVG-TxNqeflFknBL1JlHfa5Y2Kww")

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(
        f'{text_2}{message}')
    assesment_2 = response.text
    c = canvas.Canvas("diabetic_retinology.pdf", pagesize=letter)
    heading_text = "Main Heading"
    heading_font_size = 24
    heading_width = c.stringWidth(heading_text, "Helvetica", heading_font_size)
    heading_x = (letter[0] - heading_width) / 2  # Centering the heading horizontally
    heading_y = 750  # Adjust vertical position as needed
    c.setFont("Helvetica", heading_font_size)
    c.drawString(heading_x, heading_y, heading_text)
    textobject = c.beginText()
    textobject.setTextOrigin(50, 750)  # Adjust position as needed
    textobject.setFont("Helvetica", 12)  # Adjust font and size as needed
    for line in assesment_2.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()


    c = canvas.Canvas("kidney_disease_detection.pdf", pagesize=letter)
    textobject = c.beginText()
    textobject.setTextOrigin(50, 750)  # Adjust position as needed
    textobject.setFont("Helvetica", 12)  # Adjust font and size as needed
    for line in text_1.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()



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

#

if __name__ == '__main__':
    app.run(port=5000)

