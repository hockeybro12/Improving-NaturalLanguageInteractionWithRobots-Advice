'''
This file provides code to train a model to understand restrictive advice. The model takes a random coordinate and an advice sentence as input, and must output a positive prediction if the coordinate is in the advice region, else negative. As all the data is manually generated, you can have unlimited training examples. In the future, the advice could be collected from humans.

If you set FLAGS.self_generated_advice to True, it will generate advice and learn the model to learn the significantly more regions that are used in input-specific self-generated advice.

The code flows top to down. The only variables which must be changed are at the top. You must define where the model must be saved. 
'''

from __future__ import division
from absl import flags
import os,random,sys
sys.path.append(".")

## Model Imports
import tensorflow as tf
tf.set_random_seed(20160905)
import numpy as np
import argparse
np.set_printoptions(threshold=np.nan)
import pandas as pandas
import math
import time
import random
import nltk
from nltk.tokenize import word_tokenize
from random import shuffle
from TFLibraries.Layer import Layers
Layer = Layers()

# parameters to edit
# where to save the entire model and just the advice portion that will be transfered to the end-to-end model in the BaselineModelAdvice.py file

flags.DEFINE_string("general_save_path", default="savedModels/pre_trained_advice", help='folder to save the entire model')
flags.DEFINE_string("model_save_path", default='savedModels/pre_trained_advice' + '/model.ckpt', help='File to save the entire model')
flags.DEFINE_string("advice_model_save_path", default='savedModels/pre_trained_advice' + "_advice" + "/model.ckpt", help='folder to save just the advice part')

flags.DEFINE_bool("self_generated_advice", default=False, help="True if you are trying to understand input specific self-generated advice which has many more regions.")
flags.DEFINE_string("save_tokens_path", default="saved_tokens/tokens.npy", help="Where to save the tokens of the advice to use in the end-to-end model.")

flags.DEFINE_integer("num_epochs", default=50, help="how many epochs to run")
flags.DEFINE_integer("advice_embedding_size", default=100, help="size of the embedding that understands the advice")
flags.DEFINE_integer("advice_hidden_layer_size", default=256, help="size of the fully connected layer after advice LSTM")
flags.DEFINE_float("learning_rate", default=0.0001, help="learning rate")
flags.DEFINE_float("dropout_prob", default=0.25, help="dropout probability")
flags.DEFINE_float("gradient_clip_threshold", default=5.0, help="gradient clip threshold")
flags.DEFINE_integer("batch_size", default=128, help="batch size")
flags.DEFINE_integer("random_seed_value", default=20160501, help="random seed")
flags.DEFINE_integer("num_data_points", default=1200000, help="how many data points to generate. We can generate unlimited examples for the pre-trained models, so this number controls it.")
flags.DEFINE_integer("output_dims", default=2, help="How many output nodes in the last layer. 2 for binary classification.")
flags.DEFINE_integer("maxadvicelength", default=40, help="Maximum how long the advice text can be")
flags.DEFINE_integer("num_layers", default=1, help="How many LSTM layers")
flags.DEFINE_integer("fc_column_size", default=100, help="Size of the FC layers that change the dimension of the random input coordinate.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

# keep track of the maximum region you have learned for input specific model self-generated advice
global max_region 
max_region = 0


# the advice we will provide to the system. In this case we are simulating the advice by filling regions as placeholders into pre-defined sentences. In the future, this advice could be gotten from the human operator.
advice_list = ["The block's %s and %s location is in the %s",
            "The box is in the %s",
            "In the %s"]


def tokenize_advice(generated_advice_list):
    '''create the tokens dictionary for all the advice''' 

    # generate all possible advice strings
    possible_advice = ' '.join([a for a in generated_advice_list])

    # remove all the placeholders
    possible_advice = possible_advice.replace("%.2f", "")
    possible_advice = possible_advice.replace("%s", "")
    possible_advice = possible_advice.replace("%d", "")
    # tokenize the words
    tokens = list(set(word_tokenize(possible_advice))) + ["block's"]
    tokens_dictionary = {token: idx + 1 for idx, token in enumerate(set(tokens))}
    return tokens_dictionary


def determine_advice_region(x_coordinate, y_coordinate):
    ''' determine which advice region the coordinate falls in '''
    # 4 quadrants
    # upper left
    if x_coordinate < 0 and y_coordinate > 0:
        region_number = 0
    elif x_coordinate < 0 and y_coordinate <= 0:
        # lower left
        region_number = 1
    elif x_coordinate >= 0 and y_coordinate > 0:
        # upper right
        region_number = 2
    elif x_coordinate >= 0 and y_coordinate <= 0:
        # lower right
        region_number = 3

    return region_number


def generate_advice_given_target(advice_list, x_coordinate, z_coordinate):
    '''input-specific self-generated advice: generate the advice sentences in this case.'''

    # go through each and every cell and see which one we fit into 

    distance_threshold = 0.01

    x_left_bound = -1.0
    x_right_bound = 0.0
    x_region_counter = 0

    z_left_bound = -1.0
    z_right_bound = 0.0
    z_region_counter = 0 

    # this variable just helps us keep track of the max region, not practically necessary
    global max_region

    # find the left and right bounds for the coordinate. determine what to call the advice region based on this.
    while(True):
        if x_coordinate >= x_left_bound and x_coordinate <= x_right_bound:
            break
        else:
            x_left_bound = x_left_bound + distance_threshold
            x_right_bound = x_right_bound + distance_threshold
            x_region_counter = x_region_counter + 1

    # find the left and right bounds for the coordinate. determine what to call the advice region based on this.
    while(True):
        if z_coordinate >= z_left_bound and z_coordinate <= z_right_bound:
            break
        else:
            z_left_bound = z_left_bound + distance_threshold
            z_right_bound = z_right_bound + distance_threshold
            z_region_counter = z_region_counter + 1


    max_region = max(x_region_counter, max_region)

    # generate the advice. 1/3 of the time have a negative example
    binary_prediction = 1
    if (np.random.randint(0, 3) == 1):
        # use this is as a negative example
        binary_prediction = 0
        x_random_region = np.random.randint(100)
        z_random_region = np.random.randint(100) 
        advice_sentence = "XRegion " + str(x_random_region) + " ZRegion " + str(z_random_region)
        # if the random one we generated happened to be correct, then change the binary prediction
        if x_random_region == x_region_counter and z_random_region == z_region_counter:
            binary_prediction = 1
    else:
        advice_sentence = "XRegion " + str(x_region_counter) + " ZRegion " + str(z_region_counter) 

    return advice_sentence, binary_prediction


def generate_advice(advice_list, random_coordinate, maxdeviation, training=True):
    '''generate the advice text given a random coordinate'''

    # upper left, lower left, upper right, lower right
    coordinate_string_dict = {0: "upper left", 1: "lower left", 2: "upper right", 3: "lower right"}
    coordinate_string_dict_2 = {0: "north western region", 1: "south western region", 2:"north eastern region", 3: "south eastern region"}
    
    # randomly choose one of the above placeholders to fill in 
    possibleDictionariesList = [coordinate_string_dict, coordinate_string_dict_2]
    chosenDictionary = random.choice(possibleDictionariesList)

    # sometimes generate the correct advice region, sometimes make the advice region random. This allows us to have positive and negative binary predictions and a somewhat balanced data
    probability_correct_number = np.random.randint(0, 7)
    if probability_correct_number == 0 or probability_correct_number == 1 or probability_correct_number == 2:
        quadrant_number = determine_advice_region(random_coordinate[0], random_coordinate[2])
    else:
        quadrant_number = np.random.randint(0, 4)

    word_to_fill = chosenDictionary[quadrant_number]

    # fill in the region word into the advice sentences
    advice_0 = advice_list[0] % ("x", "z", word_to_fill)
    advice_1 = advice_list[1] % (word_to_fill)
    advice_2 = advice_list[2] % (word_to_fill)

    # randomly choose one advice sentence
    possible_advice_list = [advice_0, advice_1, advice_2]
    chosen_advice = random.choice(possible_advice_list)

    # if the advice text matches the input coordinate, the prediction should be positive else negative
    binary_prediction = 1 if determine_advice_region(random_coordinate[0], random_coordinate[2]) == quadrant_number else 0

    return chosen_advice, binary_prediction


def tokenize_current_advice(given_advice, tokenized_advice_dict):
    '''tokenize an advice sentence based on the tokens dictionary'''
    given_advice = given_advice.replace(",", "")

    advice_vector = np.zeros(FLAGS.maxadvicelength)
    advice_tokens = given_advice.split(" ")
    for i in range(len(advice_tokens)):
        advice_vector[i] = tokenized_advice_dict[advice_tokens[i]]

    return advice_vector, len(advice_tokens)

def mergeLists(list_numbers, list_letters):
    '''helper function to merge two lists'''
    answer = []
    i = 0
    for i in range(len(list_letters)):
        answer.append(round(list_letters[i], 2))
        answer.append(round(list_numbers[i], 2))
    return answer

def generateNumbersData(data_points):
    '''generate a bunch of random coordinates'''
    coords = np.random.uniform(-1.00, 1.00, 3 * data_points)
    return coords

# model definition starts here:

# helper functions:
def build_column(x, input_size, column_size):
    '''helper function to take input and run a FC layer on it with custom initialized weights and bias'''
    w = tf.Variable(tf.random_normal([input_size, column_size]))
    b = tf.Variable(tf.random_normal([column_size]))
    processing1 = tf.nn.relu(tf.matmul(x, w) + b)
    return processing1
def lstm_cell(hidden_size):
    '''lstm cell'''
    lstmCell = tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer(seed=FLAGS.random_seed_value))
    lstmCell = tf.contrib.rnn.DropoutWrapper(lstmCell, output_keep_prob=FLAGS.dropout_prob)
    return lstmCell

# this is placeholders for the random input coordinate
x_data_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size, 1], name="x_data")
y_data_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size, 1], name="y_data")
z_data_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size, 1], name="z_data")
# advice and labels placeholders
advice_data = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.maxadvicelength], name="advice_data")
advice_lengths = tf.placeholder(tf.int32, [FLAGS.batch_size], name="advice_lengths")
y_labels_placeholder = tf.placeholder(tf.int64, [FLAGS.batch_size], name="y_labels_placeholder")

# create the LSTM
advice_LSTM = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in FLAGS.num_layers * [FLAGS.advice_hidden_layer_size]], state_is_tuple=True)

# embed the advice text
with tf.variable_scope("advice_embeddings"):
    advice_embeddings = tf.Variable(tf.random_uniform([200, FLAGS.advice_embedding_size], -1, -1, seed=20160503))
advice_rnn_inputs = tf.nn.embedding_lookup(advice_embeddings, advice_data)

# run the LSTM
with tf.variable_scope("advice_LSTM"):
    advice_outputs, advice_fstate = tf.nn.dynamic_rnn(cell=advice_LSTM, 
        inputs=advice_rnn_inputs, 
        sequence_length=advice_lengths, 
        dtype=tf.float32, 
        time_major=False)
# add a fully connected layer after the LSTM
with tf.name_scope("lstm_output_layer"):
    lstm_output_layer = build_column(tf.concat([f.h for f in advice_fstate], 1), FLAGS.advice_hidden_layer_size * FLAGS.num_layers, FLAGS.fc_column_size)

with tf.name_scope("fc_inputs"):
    # have fully connected layers to map them the random input coordinates into the same dimension as the LSTM output layer from above
    x_output_layer = build_column(x_data_placeholder, 1, FLAGS.fc_column_size)
    y_output_layer = build_column(y_data_placeholder, 1, FLAGS.fc_column_size)
    z_output_layer = build_column(z_data_placeholder, 1, FLAGS.fc_column_size)

    total_output_layer = x_output_layer + y_output_layer + z_output_layer + lstm_output_layer

with tf.name_scope("output-3-weight"):
    output_layer = Layer.W(FLAGS.advice_hidden_layer_size, FLAGS.output_dims, 'OutputLayer2')
with tf.name_scope("output-3-bias"):
    output_bias = Layer.b(FLAGS.output_dims, 'OutputBias2')

with tf.name_scope("fc_layer"):
    fc_1 = tf.contrib.layers.fully_connected(total_output_layer, num_outputs=FLAGS.advice_hidden_layer_size, activation_fn=tf.nn.relu)

with tf.name_scope("logits"):
    logits = tf.matmul(fc_1, output_layer) + output_bias
# do a simple binary prediction
with tf.name_scope("softmax"):
    result_prediction = tf.nn.softmax(logits)

# which variables to save
advice_saved_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "advice_LSTM")
advice_saved_variables = advice_saved_variables + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "advice_embeddings")
advice_saved_variables = advice_saved_variables + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "lstm_output_layer")

training_variables = advice_saved_variables + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fully_connected")
training_variables = training_variables + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "output-3-weight")
training_variables = training_variables + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "output-3-bias")

# loss function and optimizer and gradient clipping
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_labels_placeholder, logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.gradient_clip_threshold)
    optimize = optimizer.apply_gradients(zip(gradients, variables))

# compute the accuracy
with tf.name_scope("accuracy_calculation"):
    current_predictions = tf.argmax(result_prediction, 1)
    correct_prediction = tf.equal(tf.argmax(result_prediction, 1), y_labels_placeholder)
    result_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# save the model to be used later in the other network
advice_saver = tf.train.Saver(advice_saved_variables)

# save the model in general 
saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())

def generate_advice_list(random_coords, batch_size, advice_list, training=True):
    '''generate the advice'''

    generated_advice_list = []
    binaryPredictionList = []
    currentXList = []
    currentYList = []
    currentZList = []


    # get coordinates. 
    for i in np.arange(0, len(random_coords), step=6):
        current_x_coord = round(random_coords[i], 2)
        current_y_coord = round(random_coords[i+1], 2)
        current_z_coord = round(random_coords[i+2], 2)
        current_target = [current_x_coord, current_y_coord, current_z_coord]

    
        if FLAGS.self_generated_advice:
            generated_advice, generatedPrediction = generate_advice_given_target(advice_list, current_target[0], current_target[2])
        else:
            generated_advice, generatedPrediction = generate_advice(advice_list, current_target, maxdeviation=0.25, training=training)

        generated_advice_list.append(generated_advice)
        binaryPredictionList.append(generatedPrediction)

        currentXList.append([current_x_coord])
        currentYList.append([current_y_coord])
        currentZList.append([current_z_coord])

    return generated_advice_list, binaryPredictionList, currentXList, currentYList, currentZList

def generateInputData(generated_advice_list, binaryPredictionList, currentXList, currentYList, currentZList, batch_size, advice_tokens_dictionary):
    '''generate the feed dict that can be passed into the model'''

    # placeholders
    feedDictionaryList = []
    x_input_vector = np.zeros((FLAGS.batch_size, 1), np.float32)
    y_input_vector = np.zeros((FLAGS.batch_size, 1), np.float32)
    z_input_vector = np.zeros((FLAGS.batch_size, 1), np.float32)
    true_labels = np.zeros((FLAGS.batch_size), np.float)
    advice_input_vector = np.zeros((FLAGS.batch_size, FLAGS.maxadvicelength), np.int32)
    advice_size_length = []

    batch_counter = 0

    for i in range(len(generated_advice_list)):
        current_x = currentXList[i][0]
        current_y = currentYList[i][0]
        current_z = currentZList[i][0]

        # fill in the random input coordinate
        x_input_vector[batch_counter] = current_x
        y_input_vector[batch_counter] = current_y
        z_input_vector[batch_counter] = current_z

        # fill the labels
        true_labels[batch_counter] = binaryPredictionList[i]


        # tokenize the advice sentence
        current_advice, current_advice_length = tokenize_current_advice(generated_advice_list[i], advice_tokens_dictionary)

        # keep track of the length and set up the input vector for the text
        advice_size_length.append(current_advice_length)
        for j in range(len(current_advice)):
            advice_input_vector[batch_counter, j] = current_advice[j]


        # if we have seen all the examples that will be part of this batch, then create the feed dictionary
        if batch_counter == (FLAGS.batch_size - 1):
            feed_dict = {}
            feed_dict = {advice_data: advice_input_vector, advice_lengths: advice_size_length, x_data_placeholder: x_input_vector, y_labels_placeholder: true_labels, y_data_placeholder: y_input_vector, z_data_placeholder: z_input_vector}
            feedDictionaryList.append(feed_dict.copy())

            # reset all our vectors / counters
            batch_counter = 0
            x_input_vector = np.zeros((FLAGS.batch_size, 1), np.float32)
            y_input_vector = np.zeros((FLAGS.batch_size, 1), np.float32)
            z_input_vector = np.zeros((FLAGS.batch_size, 1), np.float32)
            true_labels = np.zeros((FLAGS.batch_size), np.float)
            advice_input_vector = np.zeros((FLAGS.batch_size, FLAGS.maxadvicelength), np.int32)
            advice_size_length = []
        else:
            # increment batch counter, move to next example
            batch_counter = batch_counter + 1

    return feedDictionaryList

def train_test_model(sess, feedDictionary, training=True):
    '''function that does the training and testing'''

    shuffle(feedDictionary)

    total_loss = []
    predictions = []

    for feed_dict in feedDictionary:

        if training == True:
            _, current_loss = sess.run([optimize, loss], feed_dict=feed_dict)
            total_loss.append(current_loss)

        else:
            predictions.append(sess.run(result_accuracy, feed_dict))

    if training == True:
        return np.mean(total_loss)
    else:
        return predictions

# generate data
random_coordinate_data = generateNumbersData(FLAGS.num_data_points)
random_coordinate_data = np.asarray(random_coordinate_data)

# partition the data into train, validate, test
data_length = len(random_coordinate_data)
indices = range(data_length)
training_index, validation_index, test_index = indices[:int(.6 * data_length)], indices[int(.6 * data_length):int(.8 * data_length)], indices[int(.8 * data_length):]

training_x_data = random_coordinate_data[training_index]
validation_x_data = random_coordinate_data[validation_index]
test_x_data = random_coordinate_data[test_index]

# generate the advice
train_advice_list, train_prediction_list, trainXList, trainYList, trainZList = generate_advice_list(training_x_data, FLAGS.batch_size, advice_list)
validate_advice_list, validate_prediction_list, validateXList, validateYList, validateZList = generate_advice_list(validation_x_data, FLAGS.batch_size, advice_list)
test_advice_list, test_prediction_list, testXList, testYList, testZList = generate_advice_list(test_x_data, FLAGS.batch_size, advice_list)

# build the tokens dictionary with the train advice which should have all possible variations of text that could be in advice sentences
advice_tokens_dictionary = tokenize_advice(train_advice_list)

# where to save the advice tokens. path defined at the top
np.save(FLAGS.save_tokens_path, advice_tokens_dictionary)

# generate the feed dicts that will be passed to sess.run for the train, validation, and test data
feedDictionaryListTrain = generateInputData(train_advice_list, train_prediction_list, trainXList, trainYList, trainZList, FLAGS.batch_size, advice_tokens_dictionary)
feedDictionaryListValidate = generateInputData(validate_advice_list, validate_prediction_list, validateXList, validateYList, validateZList, FLAGS.batch_size, advice_tokens_dictionary)
feedDictionaryListTest = generateInputData(test_advice_list, test_prediction_list, testXList, testYList, testZList, FLAGS.batch_size, advice_tokens_dictionary)

# save the best accuracy
best_train_accuracy = 0.0
best_test_accuracy = 0.0
best_validation_accuracy = 0.0

for epoch in range(FLAGS.num_epochs):

    start_time = time.time()

    # train model and compute accuracy
    trainLoss = train_test_model(session, feedDictionaryListTrain)
    # train_predictions is actually the accuracy for each batch
    train_predictions = train_test_model(session, feedDictionaryListTrain, training=False)
    train_accuracy = float(sum(train_predictions) / len(train_predictions))
    best_train_accuracy = max(best_train_accuracy, train_accuracy)

    print("Epoch " + str(epoch))
    print("Train accuracy:")
    print(train_accuracy)

    validate_predictions = train_test_model(session, feedDictionaryListValidate, training=False)
    validate_accuracy = float(sum(validate_predictions) / len(validate_predictions))

    test_predictions = train_test_model(session, feedDictionaryListTest, training=False)
    test_accuracy = float(sum(test_predictions) / len(test_predictions))
    # if we have the best accuracy, save
    if test_accuracy > best_test_accuracy:
        # save the model
        save_path = saver.save(session, FLAGS.model_save_path)
        advice_save_path = advice_saver.save(session, FLAGS.advice_model_save_path)
        print("saving models at " + save_path)
        print("saving advice portion at " + advice_save_path) 
        best_validation_accuracy = validate_accuracy
        best_test_accuracy = test_accuracy

    elapsed_time = time.time() - start_time

    print("Validation accuracy:")
    print(validate_accuracy)
    print("Test accuracy:")
    print(test_accuracy)

    print("Best train average: " + str(best_train_accuracy))
    print("Best validation_average: " + str(best_validation_accuracy))
    print("Best test average: " + str(best_test_accuracy))

    print("Elapsed time:")
    print(elapsed_time)
    print("")

# print the best results
print("Best train average: " + str(best_train_accuracy))
print("Best validation_average: " + str(best_validation_accuracy))
print("Best test average: " + str(best_test_accuracy))