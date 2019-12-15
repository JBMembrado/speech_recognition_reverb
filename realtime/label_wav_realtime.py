"""Runs a trained audio graph against a WAVE file and reports the results for real time classification"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf


FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.io.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
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
      print('%s (score = %.5f)' % (human_string, score))
    return top_k


def label_wav(wav_data, labels, graph, input_name='wav_data:0', output_name='labels_softmax:0', how_many_labels=3):
  """Loads the model and labels, and runs the inference to print predictions."""

  if not labels or not tf.io.gfile.exists(labels):
    tf.compat.v1.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.io.gfile.exists(graph):
    tf.compat.v1.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  return run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)

