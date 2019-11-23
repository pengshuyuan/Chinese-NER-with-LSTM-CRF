# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob


class Model(object):
    def __init__(self, config):
        # super parameters
        self.config = config
        self.lr = config['lr']
        self.char_dim = config['char_dim']
        self.seg_dim = config['seg_dim']
        self.lstm_dim = config['lstm_dim']

        self.num_tags = config['num_tags']
        self.num_chars = config['num_chars']
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 =tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # placeholder
        self.char_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='CharInputs') # batch_size * LenSentence
        self.seg_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='SgeInputs')
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='Targets')
        self.dropout = tf.placeholder(dtype=tf.float32, name='Dropout')

        # lengths, batch_size, num_steps
        used = tf.sign(tf.abs(self.char_input))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_input[0])
        self.num_steps = tf.shape(self.char_input[1])

        # Net Structure
        embedding_output = self.embedding_layer(char_inputs=self.char_input, seg_inputs=self.seg_input, config=config)
        embedding_dropout = tf.nn.dropout(embedding_output, self.dropout)
        lstm_output = self.biLSTM_layer(lstm_inputs=embedding_dropout, lstm_dim=self.lstm_dim, lengths=self.lengths)
        self.logits = self.project_layer(lstm_outputs=lstm_output)
        self.loss = self.loss_layer(project_logits=self.logits, lengths=self.lengths)

        # optimizer
        opt_kind = self.config['optimizer']
        if opt_kind == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(self.lr)
        elif opt_kind == 'adam':
            self.opt = tf.train.AdamOptimizer(self.lr)
        elif opt_kind == 'adgrad':
            self.opt = tf.train.AdagradOptimizer(self.lr)
        else:
            raise KeyError

        # apply grad clip to avoid gradient explosion
        grads_vars = self.opt.compute_gradients(self.loss)
        capped_grads_vars = [[tf.clip_by_value(g, -self.config['clip'], self.config['clip']), v]
                            for g, v in grads_vars]
        self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # model saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        embedding = list()
        with tf.variable_scope('char_embedding' if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name='char_embedding',
                                               shape=[self.num_chars, self.char_dim],
                                               initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if self.seg_dim:
                with tf.variable_scope('seg_embedding'), tf.device('/cpu: 0'):
                    self.seg_lookup = tf.get_variable(name='seg_embedding',
                                                      shape=[self.num_segs, self.seg_dim],
                                                      initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))

            return tf.concat(embedding, axis=-1)


    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        with tf.variable_scope('char_lstm' if not name else name):
            lstm_cell = {}
            for direction in ['forward', 'backward']:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell['forward'],
                lstm_cell('backward'),
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths
            )

            return  tf.concat(outputs, axis=2)


    def project_layer(self, lstm_outputs, name=None):
        with tf.variable_scope('project' if not name else name):
            with tf.variable_scope('hidden'):
                W = tf.get_variable(name='W', shape=[self.lstm_dim*2, self.lstm_dim], dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable(name='b', shape=[self.lstm_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
                lstm_outputs_reshape = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden_output = tf.tanh(tf.nn.xw_plus_b(lstm_outputs_reshape, W, b))
            with tf.variable_scope('logits'):
                W = tf.get_variable(name='W', shape=[self.lstm_dim, self.num_tags], dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable(name='b', shape=[self.num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden_output, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])


    def loss_layer(self, project_logits, lengths, name=None):
        with tf.variable_scope('crf_loss' if not name else name):
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags])],
                 tf.zeros(shape=[self.batch_size, 1, 1]),
                 axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat([tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                name='transitions',
                shape=[self.num_tags+1, self.num_tags+1],
                initializer=self.initializer)


    def create_feed_dict(self, is_train, batch):


    def run_step(self, sess, is_train, batch):


    def decode(self, logits, lengths, matrix):


    def evaluate(self, sess, data_manager, id_to_tag):


    def evaluate_line(self, sess, inputs, id_to_tag):
