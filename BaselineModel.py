'''
This file describes our re-implementation of the best model proposed by Bisk et. al in their work "Natural Language Communication with Robots." All our advice models are built on top of this model.  

The code flows top down and the parameters are at the top.
'''

from absl import flags

import os,random,sys
sys.path.append(".")

## Model Imports
import tensorflow as tf
tf.set_random_seed(20160905)
random.seed(20160427)
import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pandas
import math
import time
import argparse
from TFLibraries.Layer import Layers
Layer = Layers()

# these are the main parameters you should edit
flags.DEFINE_integer("target", default=1, help="FLAGS.target is 1 if source coordinate prediction, else 2 if target prediction")
flags.DEFINE_string("model_save_path", default='savedModels/default_model/model.ckpt', help="Where to save the trained model.")
flags.DEFINE_string("train_file", default='data/STxyz_Blank/Train.mat', help='Where the training data mat file is located.')
flags.DEFINE_string("dev_file", default='data/STxyz_Blank/Dev.mat', help='Where the dev data mat file is located.')
flags.DEFINE_string("test_file", default='data/STxyz_Blank/Test.mat', help='Where the test data mat file is located.')

# these parameters can be left as default to train the model and achieve the performance we report
flags.DEFINE_integer("hidden_layer_size", default=256, help="hidden layer size for the LSTM and FC Layers.")
flags.DEFINE_integer("word_embedding_size", default=256, help="Size of the word embedding layer.")
flags.DEFINE_integer("epochs", default=60, help="How many FLAGS.epochs to run.")
flags.DEFINE_float("learning_rate", default=0.001, help="Learning rate. No decay implemented.")
flags.DEFINE_float("gradient_clip_threshold", default=5.0, help="When to do gradient clipping.")
flags.DEFINE_integer("maxlength", default=105, help="Maximum sentence length")
flags.DEFINE_integer("world_length", default=20, help="Length of the world (grid) array")
flags.DEFINE_integer("world_size", default=3, help="Width of the world (grid) array")
flags.DEFINE_integer("batch_size", default=9, help="How many examples per batch. Set to 9 because there are 9 different sentences per world configuration.")
flags.DEFINE_integer("ndirs", default=9, help="Dimension for the number of directions prediction FC layer.")
flags.DEFINE_integer("nblocks", default=20, help="Dimensions for the number of blocks prediction FC layer.")
flags.DEFINE_integer("ndims", default=3, help="How many output dimensions. Should be 3 due to 3 coordinates.")
flags.DEFINE_bool("performing_analysis", default=False, help="Whether we are performing analysis or not. If true, don't save the model and don't run training.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)


def load_data(train_file, dev_file, test_file):
    '''Load the data given the input files'''

    # dictionaries to store all the data
    training = {}
    training_labels = {}
    training_lens = {}
    development = {}
    development_labels = {}
    development_lens = {}
    testing = {}
    testing_labels = {}
    testing_lens = {}

    # if FLAGS.target = 1, range = 1 to 3
    range_start = 0
    range_end = 3
    if FLAGS.target == 2:
        range_start = 3
        range_end = 6

    xvocab = 0

    # Load the Data
    print("Reading the data files...")
    # A minibatch consists of a FLAGS.target location, a world, and 9 sentences that share the same FLAGS.target/world.

    fileList = [FLAGS.train_file, FLAGS.dev_file, FLAGS.test_file]
    # this list will hold train, dev, test files and all of their individual minibatches
    all_data = []

    for k in range(0, 3):
        df = pandas.DataFrame([line.strip().split() for line in open(fileList[k], 'r')])
        df.fillna("", inplace=True)
        data = np.array(df)


        # A minibatch consists of a FLAGS.target location, a world, 
        # and 9 sentences that share the same FLAGS.target/world.
        minibatches = []

        # go from 1 to d in steps of 9
        # 11870
        for i1 in range(0, data.shape[0] - 1, 9):
            # go through each individual one
            FLAGS.target = np.reshape(np.asarray(data[i1, range_start:range_end], dtype=np.float), (3, 1))
            world = np.reshape(np.asarray(data[i1,6:66], dtype=np.float), (3,20), order='F')
            sentences = []
            for i in range(i1, i1+9):
                # get in batches of 60
                #@assert FLAGS.target == np.reshape(np.asarray(data[i1, range_start:range_end], dtype=np.float32), (3, 1))
                # @assert world == np.reshape(np.asarray(data[i1,6:66]), (3,20))
                sent = []
                # maybe this is 67
                for j in range (66, len(data[2])):
                    # if "", break, else add to the sentence
                    if data[i, j] == "":
                        break
                    sent.append(data[i,j])
                    if int(data[i, j]) > xvocab:
                        if fileList[k] == FLAGS.train_file:
                            xvocab = int(data[i, j])
                        
                sentences.append(sent)
                    
            minibatches.append((FLAGS.target, world, sentences))

        all_data.append(minibatches)

    return all_data, xvocab

print("Initializing the model...")

# load the data
all_data, xvocab = load_data(FLAGS.train_file, FLAGS.dev_file, FLAGS.test_file)

lastloss = bestloss = sys.maxint
# train is all_data[0], dev is all_data[1], test is all_data[2]

# FLAGS.maxlength = 83 since we can possible have an 83 word sentence and must pass in one by one
input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.maxlength])
# store how many sequences to go until we have to update the gradients
lengths = tf.placeholder(tf.int32, [FLAGS.batch_size])
# correct outputs is a matrix of the correct x,y,z coordinates for each element in the batch
labels = tf.placeholder(tf.float32, [3, FLAGS.batch_size])
with tf.name_scope("dropout_placeholer"):
    dropout_prob_placeholder = tf.placeholder_with_default(1.0, shape=())

embeddings = tf.Variable(tf.random_uniform([xvocab, FLAGS.word_embedding_size], -1, 1, seed=20160503))

# RNN architecture
multicells = 1

# the lstm cell
lstm = tf.contrib.rnn.LSTMCell(FLAGS.hidden_layer_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer(seed=20160501))
# dropout cell
lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob_placeholder)
# create a cell of the LSTM cell 
lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm] * multicells, state_is_tuple=True)

# we have two softmax, one of size FLAGS.nblocks, another of size FLAGS.ndirs
output_layer = {}
with tf.name_scope("output-20-weight"):
    output_layer[0] = Layer.W(1 * FLAGS.hidden_layer_size, FLAGS.nblocks, 'OutputLayer')
with tf.name_scope("output-10-weight"):
    output_layer[1] = Layer.W(1 * FLAGS.hidden_layer_size, FLAGS.ndirs, 'OutputLayer2')
# # add bias to them, not sure if needed
output_bias = {}
with tf.name_scope("output-20-bias"):
    output_bias[0] = Layer.b(FLAGS.nblocks, 'OutputBias')
with tf.name_scope("output-10-bias"):
    output_bias[1] = Layer.b(FLAGS.ndirs, 'OutputBias2')


# inputs
rnn_inputs = tf.nn.embedding_lookup(embeddings, input_data)
# make the RNN graph
with tf.variable_scope("lstm0"):
    # create the rnn graph at run time
    # sequence length allows us to input variable lengths
    # tensorflow returns zero vectors for states and outputs only after the sequence length.
    outputs, fstate = tf.nn.dynamic_rnn(cell=lstm, inputs=rnn_inputs,
                                        sequence_length=lengths, 
                                        dtype=tf.float32, time_major=False)


logits = {}
with tf.name_scope("output1-20-compute"):
    logits[0] = tf.matmul((fstate[0].h), output_layer[0]) + output_bias[0]

with tf.name_scope("output2-9-compute"):
    logits[1] = tf.matmul((fstate[0].h), output_layer[1]) + output_bias[1]

# FLAGS.nblocks output
with tf.name_scope("softmax-20"):
    refblock = tf.nn.softmax(logits[0])
# FLAGS.ndirs output
with tf.name_scope("softmax-9"):
    direction = tf.nn.softmax(logits[1])

world_placeholder = tf.placeholder(tf.float32, [FLAGS.world_size, FLAGS.world_length])

# multiply the world by the softmax output of size FLAGS.nblocks (20)
refxyz = tf.matmul(world_placeholder, tf.transpose(refblock))

with tf.name_scope("Weights_9_to_3_Dims"):
    output_dimensions = Layer.W(FLAGS.ndirs, FLAGS.ndims, name='OffsetWeights')
    offset = tf.matmul(direction, output_dimensions)

# add these results together to get a matrix of size (3, 9)
with tf.name_scope("resulting_coordinate"):
    result = refxyz + tf.transpose(offset)   


# Learning
with tf.name_scope("regular_optimizer"):
    loss = tf.reduce_mean(tf.squared_difference(result, labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.gradient_clip_threshold)
    optimize = optimizer.apply_gradients(zip(gradients, variables))

with tf.name_scope("correct_prediction"):
    # distance from the coordinates normalized by the block length
    correct_prediction = tf.reduce_sum([(tf.sqrt(tf.reduce_sum([(result[:, j][i] - labels[:, j][i])**2 for i in range(3) ]))/0.1524) for j in range(FLAGS.batch_size)])/FLAGS.batch_size
    output_1 = ((tf.sqrt(tf.reduce_sum([(result[:, 0][i] - labels[:, 0][i])**2 for i in range(3) ]))/0.1524))
    output_2 = ((tf.sqrt(tf.reduce_sum([(result[:, 1][i] - labels[:, 1][i])**2 for i in range(3) ]))/0.1524))
    output_3 = ((tf.sqrt(tf.reduce_sum([(result[:, 2][i] - labels[:, 2][i])**2 for i in range(3) ]))/0.1524))
    output_4 = ((tf.sqrt(tf.reduce_sum([(result[:, 3][i] - labels[:, 3][i])**2 for i in range(3) ]))/0.1524))
    output_5 = ((tf.sqrt(tf.reduce_sum([(result[:, 4][i] - labels[:, 4][i])**2 for i in range(3) ]))/0.1524))
    output_6 = ((tf.sqrt(tf.reduce_sum([(result[:, 5][i] - labels[:, 5][i])**2 for i in range(3) ]))/0.1524))
    output_7 = ((tf.sqrt(tf.reduce_sum([(result[:, 6][i] - labels[:, 6][i])**2 for i in range(3) ]))/0.1524))
    output_8 = ((tf.sqrt(tf.reduce_sum([(result[:, 7][i] - labels[:, 7][i])**2 for i in range(3) ]))/0.1524))
    output_9 = ((tf.sqrt(tf.reduce_sum([(result[:, 8][i] - labels[:, 8][i])**2 for i in range(3) ]))/0.1524))


with tf.name_scope("actual_prediction"):
    actual_prediction = result[:, 1]


## Training
saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
#saver.restore(session, FLAGS.model_save_path)


# train one set of minibatches
def train_test_model(sess, minibatches, batchsize, training=True):
    ''' Train the model if training=True, test otherwise.

    Parameters: 
    sess:
        Current tensorflow session.
    minibatches:
        Minibatches of data gotten from load_data. Pass in the one for the training, validation, or test data.
    batchsize:
        Batch size.
    training:
        True if training the model, false if just evaluating the predictions.
    '''

    sumloss = numloss = 0
    y = np.zeros((3, batchsize), np.float)
    mask = np.zeros(batchsize, np.uint8)
    input_vector = np.zeros((batchsize, FLAGS.maxlength), np.int32)

    total_loss = 0.0
    predictions = []


    # passing the data through the network
    for (FLAGS.target, world, sents) in minibatches:
        if len(sents) != batchsize:
            print("Bad length, error")

        # array to store the length of each sentence in the batch
        sequenceSizeLength = []
        for k in range(len(sents)):
            sequenceSizeLength.append(len(sents[k]))

        input_vector[:] = 0
        for j in range(len(sents)):
            s = sents[j]
            for i in range(len(s)):
                input_vector[j, i] = s[i]
            
        y[:] = 0
        y += FLAGS.target

        # create the feed dict with the input data
        feed_dict = {input_data: input_vector, lengths: sequenceSizeLength, world_placeholder: world, labels: y, dropout_prob_placeholder: 0.5}

        # no dropout if not training
        if training == False:
            feed_dict = {input_data: input_vector, lengths: sequenceSizeLength, world_placeholder: world, labels: y}

        if training == True:

            _, current_loss = sess.run([optimize, loss], feed_dict=feed_dict)
            total_loss += current_loss

        else:
            # do the evaluation
            # getting all the coordinate outputs individually is good for evaluation and also allows us to easily compute the mean/median of the entire train/test set
            resOutput1, resOutput2, resOutput3, resOutput4, resOutput5, resOutput6, resOutput7, resOutput8, resOutput9, made_prediction, resulting_values = sess.run([output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9, correct_prediction, result], feed_dict=feed_dict)
            resOutputTotal = [resOutput1, resOutput2, resOutput3, resOutput4, resOutput5, resOutput6, resOutput7, resOutput8, resOutput9]
            predictions.extend(resOutputTotal)


    if training == True:
        return total_loss
    else:
        return predictions

best_train_average = sys.maxint
best_test_average = sys.maxint
best_validation_average = sys.maxint

# do the training and evaluation
for epoch in range (FLAGS.epochs):

    start_time = time.time()

    # train if we are not in an analysis stage
    if FLAGS.performing_analysis == False:
        trainLoss = train_test_model(session, all_data[0], FLAGS.batch_size)
        print('Epoch %d: %f' % (epoch, trainLoss))

    # train dataset predictions
    predictions = train_test_model(session, all_data[0], FLAGS.batch_size, training=False)
    average0 = np.mean(predictions)
    best_train_average = min(best_train_average, average0)

    # val dataset predictions
    predictions = train_test_model(session, all_data[1], FLAGS.batch_size, training=False)
    average1 = np.mean(predictions)
    median1 = np.median(predictions)

    # test dataset predictions
    predictions = train_test_model(session, all_data[2], FLAGS.batch_size, training=False)
    average2 = np.mean(predictions)    
    median2 = np.median(predictions)

    # best performing model on the validation data should be saved and performance measured
    if average1 < best_validation_average:
        best_validation_average = average1
        best_test_average = average2
        best_test_median = median2
        if FLAGS.performing_analysis == False:
            print("Saving model at path: " + FLAGS.model_save_path)
            saver.save(session, FLAGS.model_save_path)
        best_epoch = epoch

    elapsed_time = time.time() - start_time


    # print results for each epoch
    print("Train average:")
    print(average0)
    print("Validation average:")
    print(average1)
    print("Test average:")
    print(average2)
    print("Test median")
    print(median2)
    print("Best train average: " + str(best_train_average))
    print("Best validation_average: " + str(best_validation_average))
    print("Best test average: " + str(best_test_average))
    print("Best test median: " + str(best_test_median))
    print("Elapsed time:")
    print(elapsed_time)
    print("")

# print best overall results
print("Best train average: " + str(best_train_average))
print("Best validation_average: " + str(best_validation_average))
print("Best test average: " + str(best_test_average))