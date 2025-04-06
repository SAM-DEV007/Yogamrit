from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization # type: ignore
from matplotlib import pyplot as plt

from pathlib import Path
from sklearn import metrics

import numpy as np
import tensorflow as tf

import os
import pandas as pd


if __name__ == "__main__":
    dataset = str(Path(__file__).resolve().parent / 'Data/data_v9.csv')
    model_dir = str(Path(__file__).resolve().parent / 'Model')

    model_save = os.path.join(model_dir, 'model_v9.keras')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    df = pd.read_csv(dataset, delimiter=',', header=None)
    counts = df[0].value_counts()

    print("Class distribution:")
    print(counts)
    
    x = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (18 * 2) + 1)))
    y = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = tf.keras.models.Sequential([
        Input(shape=(18 * 2,)),
        Dense(25, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(15, activation='relu'),
        BatchNormalization(),
        Dense(8, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save, verbose=1, save_weights_only=False, save_best_only=True)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback, es_callback]
    )

    val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=32)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.suptitle('Model Training')

    plt.tight_layout()

    plt.savefig(str(Path(__file__).resolve().parent / 'Model/training.png'))
    # plt.show()

    actual = y_test
    predicted = model.predict(x_test)

    predicted = np.argmax(predicted, axis=1)

    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)

    cm_display.plot()
    plt.savefig(str(Path(__file__).resolve().parent / 'Model/confusion_matrix.png'))

    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted, average='weighted')
    recall = metrics.recall_score(actual, predicted, average='weighted')
    f1 = metrics.f1_score(actual, predicted, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")