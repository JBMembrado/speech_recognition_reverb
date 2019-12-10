from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys

import tensorflow as tf
import numpy as np
import os
from datetime import datetime


class TestModel(object):
    def __init__(self, base_path, data_name):
        self.base_path = os.getcwd() + base_path
        self.data_name = data_name
        self.graph_model_path = None
        self.conv_labels_path = None
        self.list_labels = []
        self.list_paths = []
        self.how_many_labels = 3
        self.current_label = None
        self.scores = None
        self.all_scores = None

    @staticmethod
    def load_graph(filename):
        """Unpersists graph from file as default graph."""
        with tf.io.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    @staticmethod
    def load_labels(filename):
        """Read in labels, one label per line."""
        return [line.rstrip() for line in tf.io.gfile.GFile(filename)]

    def run_graph(self, wav_data, labels, input_layer_name, output_layer_name,
                  num_top_predictions):
        """Runs the audio data through the graph and prints predictions."""
        with tf.compat.v1.Session() as sess:
            # Feed the audio data as input to the graph.
            #   predictions  will contain a two-dimensional array, where one
            #   dimension represents the input image count, and the other has
            #   predictions per class
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
            predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

            # Sort to show labels in order of confidence
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]

                if human_string == self.current_label:
                    return score

            return 0

    def label_wav(self, wav, labels, graph, input_name, output_name, how_many_labels):
        """Loads the model and labels, and runs the inference to print predictions."""
        if not wav or not tf.io.gfile.exists(wav):
            tf.compat.v1.logging.fatal('Audio file does not exist %s', wav)

        if not labels or not tf.io.gfile.exists(labels):
            tf.compat.v1.logging.fatal('Labels file does not exist %s', labels)

        if not graph or not tf.io.gfile.exists(graph):
            tf.compat.v1.logging.fatal('Graph file does not exist %s', graph)

        labels_list = self.load_labels(labels)

        # load graph, which is stored in the default session
        self.load_graph(graph)

        with open(wav, 'rb') as wav_file:
            wav_data = wav_file.read()

        return self.run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)

    def read_validation_file(self):
        validation_file = self.base_path + '/validation_list.txt'
        print('Current directory is : {0}'.format(os.getcwd()))

        with open(validation_file) as file:
            for line in file:
                self.list_paths.append(line[0:-1])
                label = line.split('/')[0]
                self.list_labels.append(label)

        print('Validation file read.')
        print('Number of validation files : {0}'.format(len(self.list_labels)))
        print()

    def test_model(self, model_number):
        self.graph_model_path = self.base_path + '/../results/frozen_graph_set' + str(model_number) + '.pb'
        self.conv_labels_path = self.base_path + '/../results/conv_labels.txt'
        self.scores = np.zeros(len(self.list_labels))

        print('Starting test for model {0}'.format(model_number))

        for index_sample, path in enumerate(self.list_paths):

            if index_sample % 500 == 0:
                print('Currently at sample {0}'.format(index_sample))

            self.current_label = self.list_labels[index_sample]
            self.scores[index_sample] = self.label_wav(self.base_path + '/' + path, self.conv_labels_path,
                                                       self.graph_model_path, 'wav_data:0', 'labels_softmax:0',
                                                       self.how_many_labels)
        print('Test for model {0} done.'.format(model))
        print()

    def test_all_models(self, list_models):

        self.all_scores = np.zeros(len(list_models))
        path_to_save_scores = self.base_path + '/../results/scores_' + self.data_name + '.npy'

        for index_model, model in enumerate(list_models):
            self.test_model(model)
            self.all_scores[index_model] = np.average(self.scores)

        np.save(path_to_save_scores, self.all_scores)


start = datetime.now()

model_testing = TestModel('/tensorflow-master/tensorflow/examples/speech_commands/data', 'data_dry')
model_testing.read_validation_file()
model_testing.test_all_models([0, 2, 3, 4])

time_elapsed = datetime.now() - start
print('Time elapsed (hh:mm:ss.ms) {0}'.format(time_elapsed))
