# https://www.kaggle.com/code/kausthubkannan/ai-human-art-classification-mobilenetv2-91

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns

# Visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import class_weight

# Tensorflow
import tensorflow as tf
import tensorflow.keras.layers as lyrs
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_hub as hub
from tensorflow.keras.optimizers.legacy import Adam

root = "/kaggle/input/ai-and-human-art-classification/ai_art_classification"
IMG_SIZE = 512


def supervised_metrics(y_true, y_pred):
    """Metrics for a supervised learning model:"""
    print("Accuracy : {} %".format(accuracy_score(y_true, y_pred)*100))
    print("F1 Score : {}".format(f1_score(y_true, y_pred, average='weighted')))
    print("Recall : {}".format(recall_score(y_true, y_pred, average='weighted')))
    print("Precision : {}".format(precision_score(y_true, y_pred, average='weighted')))


def view_random_image(root_path,folder,class_folder):
    path = root_path + '/' + folder + '/' + class_folder
    rand = random.choice(os.listdir(path))
    random_image = mpimg.imread(path + '/'+rand)
    plt.imshow(random_image)
    plt.title("File Name: " + rand)


def pre_process_image(path, image_shape = 512, channels = 3, norm_factor = 255.):
    """Pre-processing the image before sending it to the model"""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels = channels)
    img = tf.image.resize(img, size = (image_shape, image_shape))
    img = tf.expand_dims(img, axis = 0)
    img = img/norm_factor
    return img


def custom_rounder(x):
    values = tf.math.round(x).numpy()
    values = np.argmax(values, axis = 1)
    if(values > 0.5):
        return 1
    return 0


def random_tester(root_path, classes, model, class_type="binary"):
    """Random class folder selection"""
    class_folder = random.choice(os.listdir(root_path))
    
    """Random file selection"""
    folder_path = root_path + '/' + class_folder + '/'
    rand = random.choice(os.listdir(folder_path))
    file_path = folder_path+'/'+rand
    random_image = mpimg.imread(file_path)

    """Prediction"""
    predicted_value = model.predict(pre_process_image(file_path)) 
    if(class_type == "binary"):
        predicted_label = classes[custom_rounder(predicted_value)]
    else:
        index = tf.math.round(predicted_value).numpy()
        index = np.argmax(index)
        predicted_label=classes[index]
        
    """Visualize"""
    plt.figure(figsize=(4, 4))
    plt.imshow(random_image)
    if(predicted_label == class_folder):
        clr = "green"
    else:
        clr = "red"
    plt.title("Prediction:" + predicted_label + "\n" + "True class: " + class_folder, color=clr)
    plt.show()


def loss_curve_plot(df):
    """ Dataframe (df) is history of the fit of the NN model
    The df consists of train and validation fit data
    """
    history = df.history
    val_accuracy = history["val_accuracy"]
    val_loss = history["val_loss"]
    train_accuracy = history["accuracy"]
    train_loss = history["loss"]
    
    """Accuracy Plot"""
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    """Loss Plot"""
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Validation loss")
    plt.title("Loss Curves")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def confusion_matrix_plot(y_true, y_pred, figsize=(30,30)):
    """"Confusion matrix for true values and predicted values"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)
    plt.figure(figsize = figsize)
    sns.heatmap(cm, annot=True, cmap="crest")


def main():
    augmentation=ImageDataGenerator(
        rescale = 1/225.,
        validation_split = 0.2
    )

    train_datagen=augmentation.flow_from_directory(
        root + "/train",
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size=  32,
        class_mode = 'categorical',
        subset = "training",
        shuffle = True
    )

    val_datagen=augmentation.flow_from_directory(
        root + "/train",
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size = 32,
        class_mode = 'categorical',
        subset = "validation",
        shuffle = False
    )

    model_base = MobileNetV2(
        input_shape = (IMG_SIZE,IMG_SIZE,3),
        include_top = False,
    )
    model_base.trainable = False

    # Transfer Learning Model
    inputs=tf.keras.Input(shape = (IMG_SIZE,IMG_SIZE,3))
    x=model_base(inputs)
    x=lyrs.GlobalAveragePooling2D()(x)
    x= lyrs.BatchNormalization()(x)
    x= lyrs.Dropout(0.5)(x)
    outputs=lyrs.Dense(2, activation = "softmax")(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.summary()

    # Fit and Train
    checkpointer = ModelCheckpoint('ai_art_classification.hdf5',verbose = 1, save_best_only = True)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr = 1e-3, decay = 1e-3), metrics = ["accuracy"])
    model_history=model.fit(x = train_datagen, 
                            steps_per_epoch = 32,
                            validation_data = val_datagen,
                            epochs = 20,
                            callbacks = [checkpointer, early_stopping])

    # Evaluate and Test
    labels=val_datagen.classes
    model=tf.keras.models.load_model("/kaggle/working/ai_art_classification.hdf5")
    y_pred=model.predict(val_datagen)
    prediction=tf.math.round(y_pred).numpy()
    prediction=prediction.argmax(axis=1)

    supervised_metrics(labels, prediction)
    confusion_matrix_plot(labels, prediction, figsize=(5,5))

    for i in range(0, 11):
        random_tester(root + "/train", os.listdir(root + "/train"), model)


if __name__ == "__main__":
    main()
