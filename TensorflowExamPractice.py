#################### QUESTION 1

# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test model button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Getting Started Question
#
# Given this data, train a neural network to match the xs to the ys
# So that a predictor for a new value of X will give a float value
# very close to the desired answer
# i.e. print(model.predict([10.0])) would give a satisfactory result
# The test infrastructure expects a trained model that accepts
# an input shape of [1]

import numpy as np

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-7.0, -3.0, 1.0, 5.0, 9.0, 13.0], dtype=float)

    # YOUR CODE HERE
    # Set random seed
    import tensorflow as tf
    tf.random.set_seed(42)

    # Create a model using the Sequential API


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
                  metrics=["mae"])

    # Fit the model
    model.fit(tf.expand_dims(xs, axis=-1), ys, epochs=600)

    print(model.predict([10.0]))

    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")


#################### QUESTION 2

# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    import numpy as np

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    TRAINING_SIZE = len(train_images)
    TEST_SIZE = len(test_images)

    print(TRAINING_SIZE)
    print(TEST_SIZE)

    # Reshape from (N, 28, 28) to (N, 28*28=784)
    # train_images = np.reshape(train_images, (TRAINING_SIZE, 784))
    # test_images = np.reshape(test_images, (TEST_SIZE, 784))

    # Convert the array to float32 as opposed to uint8
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    # print(type(train_images))
    # print(np.shape(train_images))

    # Convert the pixel values from integers between 0 and 255 to floats between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    #
    # NUM_CAT = 10
    #
    # train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CAT)
    #
    # test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CAT)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

    # We will now compile and print out a summary of our model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=15)

    # YOUR CODE HERE
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")


#################### QUESTION 3

# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ==============================================================================
#
# BASIC DATASETS QUESTION
#
# Create a classifier for the German Traffic Signs dataset that classifies 
# images of traffic signs into 43 classes.
# ==============================================================================
#
# ABOUT THE DATASET
#
# The dataset contains traffic sign boards from the streets captured into
# image files. There are 43 unique classes in total. The images are of shape
# (30,30,3).
# ==============================================================================
#
# INSTRUCTIONS
#
# We have already divided the data for training and validation.
#
# Complete the code in following functions:
# 1. preprocess()
# 2. solution_model()
#
# Your code will fail to be graded if the following criteria are not met:
# 1. The input shape of your model must be (30,30,3), because the testing
#    infrastructure expects inputs according to this specification.
# 2. The last layer of your model must be a Dense layer with 43 neurons
#    activated by softmax since this dataset has 43 classes.
#
# HINT: Your neural network must have a validation accuracy of approximately
# 0.95 or above on the normalized validation dataset for top marks.

import urllib
import zipfile

import tensorflow as tf

# This function downloads and extracts the dataset to the directory that
# contains this file.
# DO NOT CHANGE THIS CODE
# (unless you need to change https to http)
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip'
    urllib.request.urlretrieve(url, 'germantrafficsigns.zip')
    with zipfile.ZipFile('germantrafficsigns.zip', 'r') as zip_ref:
        zip_ref.extractall()

# COMPLETE THE CODE IN THIS FUNCTION
def preprocess(image, label):
    # NORMALIZE YOUR IMAGES HERE (HINT: Rescale by 1/.255)
    image = image/255.0
    label = label

    return image, label


# This function loads the data, normalizes and resizes the images, splits it into
# train and validation sets, defines the model, compiles it and finally
# trains the model. The trained model is returned from this function.

# COMPLETE THE CODE IN THIS FUNCTION.
def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()

    BATCH_SIZE = 32
    IMG_SIZE = 30

    # The following code reads the training and validation data from their
    # respective directories, resizes them into the specified image size
    # and splits them into batches. You must fill in the image_size
    # argument for both training and validation data.
    # HINT: Image size is a tuple
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train/',
        label_mode='categorical',
        image_size=  (30, 30),
        batch_size = BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation/',
        label_mode='categorical',
        image_size=  (30,30),
        batch_size = BATCH_SIZE)

    # Normalizes train and validation datasets using the preprocess() function.
    # Also makes other calls, as evident from the code, to prepare them for training.
    # Do not batch or resize the images in the dataset here since it's already
    # been done previously.

    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Code to define the model
    model = tf.keras.models.Sequential([
        # ADD LAYERS OF THE MODEL HERE

        # If you don't adhere to the instructions in the following comments,
        # tests will fail to grade your model:
        # The input layer of your model must have an input shape of
        # (30,30,3).
        # Make sure your last layer has 43 neurons activated by softmax.

        tf.keras.layers.Conv2D(64, (3,3), input_shape=(30, 30, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(43, activation=tf.nn.softmax)
    ])

    # Code to compile and train the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc']
    )

    model.fit(train_ds,
              epochs=10, # fit for 5 epochs to keep experiments quick
              validation_data=val_ds,
    )

    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

#################### QUESTION 4

# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    import pandas as pd

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE

    with open('sarcasm.json') as f:
        file = json.load(f)
        for x in file:
            sentences.append(x['headline'])
            labels.append(x['is_sarcastic'])


    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    # Setting the padding properties
    trunc_type = 'post'
    padding_type = 'post'
    # Creating padded sequences from train and test data
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(training_padded,
                        training_labels,
                        epochs=8,
                        atvalidion_data=(testing_padded, testing_labels))

    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")


#################### QUESTION 5
#release
# ==============================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative to its
# difficulty. So your Category 1 question will score significantly less than
# your Category 5 question.
#
# WARNING: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure. You do not need them to solve the question.
#
# WARNING: If you are using the GRU layer, it is advised not to use the
# recurrent_dropout argument (you can alternatively set it to 0),
# since it has not been implemented in the cuDNN kernel and may
# result in much longer training times.
#
# WARNING: Input and output shape requirements are laid down in the section 
# 'INSTRUCTIONS' below and also reiterated in code comments. 
# Please read them thoroughly. After submitting the trained model for scoring, 
# if you are receiving a score of 0 or an error, please recheck the input and 
# output shapes of the model to see if it exactly matches our requirements. 
# Grading infrastrcuture is very strict about the shape requirements. Most common 
# issues occur when the shapes are not matching our expectations.
#
# TIP: You can print the output of model.summary() to review the model
# architecture, input and output shapes of each layer.
# If you have made sure that you have matched the shape requirements
# and all the other instructions we have laid down, and still
# receive a bad score, you must work on improving your model.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ==============================================================================
#
# TIME SERIES QUESTION
#
# Build and train a neural network to predict the time indexed variable of
# the univariate US diesel prices (On - Highway) All types for the period of
# 1994 - 2021.
# Using a window of past 10 observations of 1 feature , train the model
# to predict the next 10 observations of that feature.
#
# ==============================================================================
#
# ABOUT THE DATASET
#
# Original Source:
# https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm#
#
# For the purpose of the examination we have used the Diesel (On - Highway) -
# All Types time series data for the period of 1994 - 2021 from the
# aforementioned link. The dataset has 1 time indexed feature.
# We have provided a cleaned version of the data.
#
# ==============================================================================
#
# INSTRUCTIONS
#
# Complete the code in following functions:
# 1. solution_model()
#
# You may receive a score of 0 or your code will fail to be graded if the 
# following criteria are not met:
#
# 1. Model input shape must be (BATCH_SIZE, N_PAST = 10, N_FEATURES = 1),
#    since the testing infrastructure expects a window of past N_PAST = 10
#    observations of the 1 feature to predict the next N_FUTURE = 10
#    observations of the same feature.
#
# 2. Model output shape must be (BATCH_SIZE, N_FUTURE = 10, N_FEATURES = 1)
#
# 3. The last layer of your model must be a Dense layer with 1 neuron since
#    the model is expected to predict observations of 1 feature.
#
# 4. Don't change the values of the following constants:
#    SPLIT_TIME, N_FEATURES, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT, in
#    solution_model() (See code for additional note on BATCH_SIZE).
#
# 5. Code for normalizing the data is provided - don't change it.
#    Changing the normalizing code will affect your score.
#
# 6. Code for converting the dataset into windows is provided - don't change it.
#    Changing the windowing code will affect your score.
#
# 7. Code for setting the seed is provided - don't change it.
#
# Make sure that the model architecture and input, output shapes match our
# requirements by printing model.summary() and reviewing its output.
#
# HINT: If you follow all the rules mentioned above and throughout this
# question while training your neural network, there is a possibility that a
# validation MAE of approximately 0.02 or less on the normalized validation
# dataset may fetch you top marks.


import pandas as pd
import tensorflow as tf


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


# This function is used to map the time series dataset into windows of
# features and respective targets, to prepare it for training and validation.
# The first element of the first window will be the first element of
# the dataset.
#
# Consecutive windows are constructed by shifting the starting position
# of the first window forward, one at a time (indicated by shift=1).
#
# For a window of n_past number of observations of the time
# indexed variable in the dataset, the target for the window is the next
# n_future number of observations of the variable, after the
# end of the window.

# DO NOT CHANGE THIS.
def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


# This function loads the data from the CSV file, normalizes the data and
# splits the dataset into train and validation data. It also uses
# windowed_dataset() to split the data into windows of observations and
# targets. Finally it defines, compiles and trains a neural network. This
# function returns the final trained model.

# COMPLETE THE CODE IN THIS FUNCTION
def solution_model():
    # DO NOT CHANGE THIS CODE
    # Reads the dataset.
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features of future time steps.
    N_FEATURES = len(df.columns) # DO NOT CHANGE THIS

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.8) # DO NOT CHANGE THIS
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    # DO NOT CHANGE THIS CODE
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    # DO NOT CHANGE BATCH_SIZE IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    # THE TEST WILL FAIL TO GRADE YOUR SCORE IN SUCH CASES.
    # In other cases, it is advised not to change the batch size since it
    # might affect your final scores. While setting it to a lower size
    # might not do any harm, higher sizes might affect your scores.
    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS

    # DO NOT CHANGE N_PAST, N_FUTURE, SHIFT. The tests will fail to run
    # on the server.
    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 10  # DO NOT CHANGE THIS

    # Number of future time steps which are to be predicted.
    N_FUTURE = 10  # DO NOT CHANGE THIS

    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1  # DO NOT CHANGE THIS

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    # Code to define your model.
    model = tf.keras.models.Sequential([

        # ADD YOUR LAYERS HERE.



        # If you don't follow the instructions in the following comments,
        # tests will fail to grade your code:
        # The input layer of your model must have an input shape of:
        # (BATCH_SIZE, N_PAST = 10, N_FEATURES = 1)
        # The model must have an output shape of:
        # (BATCH_SIZE, N_FUTURE = 10, N_FEATURES = 1).
        # Make sure that there are N_FEATURES = 1 neurons in the final dense
        # layer since the model predicts 1 feature.

        # HINT: Bidirectional LSTMs may help boost your score. This is only a
        # suggestion.

        # WARNING: After submitting the trained model for scoring, if you are
        # receiving a score of 0 or an error, please recheck the input and 
        # output shapes of the model to see if it exactly matches our requirements. 
        # The grading infrastructure is very strict about the shape requirements. 
        # Most common issues occur when the shapes are not matching our 
        # expectations.
        #
        # TIP: You can print the output of model.summary() to review the model
        # architecture, input and output shapes of each layer.
        # If you have made sure that you have matched the shape requirements
        # and all the other instructions we have laid down, and still
        # receive a bad score, you must work on improving your model.

        # WARNING: If you are using the GRU layer, it is advised not to use the
        # recurrent_dropout argument (you can alternatively set it to 0),
        # since it has not been implemented in the cuDNN kernel and may
        # result in much longer training times.
        #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=[N_PAST, 1])),
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=[N_PAST, 1]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Dense(N_FUTURE),
        tf.keras.layers.Dense(N_FEATURES),
    ])

    # Code to train and compile the model

    # Code to train and compile the model
    # YOUR CODE HERE

    #model.compile(loss="mae", optimizer='rm')
    model.compile(loss="mae", optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001))

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    print(model.summary())

    # Code to train and compile the model
    model.fit(train_set, epochs=100, validation_data=valid_set, callbacks=[callback])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.

if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")


# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS

#def model_forecast(model, series, window_size, batch_size):
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size))
#    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
#    forecast = model.predict(ds)
#    return forecast

# PASS THE NORMALIZED data IN THE FOLLOWING CODE

# rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
# rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]

# x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0]])
# result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

