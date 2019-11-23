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

        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="CharInputs") # dimension: BatchSize*MaxLenSentence
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")   # dimension: BatchSize*MaxLenSentence
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")        # dimension: BatchSize*MaxLenSentence
        
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)   # dimension: BatchSize，其中每个值为该批次中MaxLenSentence（字符个数）
        self.lengths = tf.cast(length, tf.int32)            # dimension: BatchSize，其中的值为每个句子的实际长度
        self.batch_size = tf.shape(self.char_inputs)[0]     # BatchSize
        self.num_steps = tf.shape(self.char_inputs)[-1]     # MaxLenSentence

        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
                                                    # embedding dimension: BatchSize*MaxLenSentence*(char_dim+seg_dim)

        # apply dropout before feed to lstm layer
        lstm_inputs = tf.nn.dropout(embedding, self.dropout) # lstm_inputs dimension: BatchSize*MaxLenSentence*(char_dim+seg_dim)
                                                             # 使输入embedding中某些元素变为0，其它没变0的元素变为原来的1/dropout大小
        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths) # lstm_outputs dimension: BatchSize*MaxLenSentence*(2*lstm_dim)

        # logits for tags
        self.logits = self.project_layer(lstm_outputs) # logits dimension: BatchSize * MaxLenSentence * num_tags

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths) # loss dimension: 标量

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion 防止梯度爆炸
            grads_vars = self.opt.compute_gradients(self.loss)   # grads_vars dimension: 可训练tf.Variable数量*2: 第一列为loss对该变量的梯度，第二列为该变量的值
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]  # 将梯度限制在[-clip, clip]范围内
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step) #

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)   #保存全部Variable，包含trainable为True和False的

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):  # embedding矩阵不是预训练的，是在网络中训练
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name="char_embedding",           
                                               shape=[self.num_chars, self.char_dim], # dimension: 字符类别数*CharEmbeddingSize
                                               initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs)) # char_inputs dimension: BatchSize*MaxLenSentence*embeddingSize
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(name="seg_embedding", 
                                                      shape=[self.num_segs, self.seg_dim], 
                                                      initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs)) # seg_inputs dimension: BatchSize*MaxLenSentence*embeddingSize
            embed = tf.concat(embedding, axis=-1)       
        return embed            # embed dimension: BatchSize*MaxLenSentence*(char_dim+seg_dim)

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]   num_steps = MaxLenSentence
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}                                                      # 定义LSTM单元
            for direction in ["forward", "backward"]:                           
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(  
                        lstm_dim,               # 输出结果维度
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)    # 若为真则输出和状态分2列输出，若为假则输出和状态在一列中输出
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(            # 定义Bi-LSTM网络
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)                            # sequence_length为BatchSize
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim], dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2]) # output dimension: (BatchSize*MaxLenSentence)*(2*lstm_dim)
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))     # hidden dimension: (BatchSize*MaxLenSentence)*lstm_dim

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags], dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, W, b)  # pred dimension: (BatchSize*MaxLenSentence)*num_tags

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags]) # dimension: BatchSize*MaxLenSentence*num_tags

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(                                 # start_logits dimension: BatchSize*1*(num_tags+1)
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), 
                 tf.zeros(shape=[self.batch_size, 1, 1])], 
                 axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1) # project_logits dimension: BatchSize*MaxLenSentence*num_tags
                                                                      # pad_logits dimension: BatchSize*MaxLenSentence*1
                                                                      # logits dimension: BatchSize*MaxLenSentence(num_tags+1)
            logits = tf.concat([start_logits, logits], axis=1)        # logits dimension: BatchSize*(1+MaxLenSentence)*(num_tags+1)
            targets = tf.concat([tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
                                                                    # targets dimension: BatchSize*(1+MaxLenSentence) 真实标签
            self.trans = tf.get_variable(                           # trans dimension: (num_targets+1)*(num_targets+1)
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,                          # 模型预测输出：BatchSize*(1+MaxLenSentence)*(num_tags+1)
                tag_indices=targets,                    # 真实标签：BatchSize*(1+MaxLenSentence)
                transition_params=self.trans,           # 转移矩阵：(num_targets+1)*(num_targets+1)
                sequence_lengths=lengths+1)             # 序列长度：BatchSize，每个值为句子长度+1
            return tf.reduce_mean(-log_likelihood)      # 标量

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch                    # batch dimension = 4 * BatchSize * MaxLenSentence
        feed_dict = {
            self.char_inputs: np.asarray(chars),        # asarray(chars)不复制chars; array(chars)复制chars
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run([self.global_step, self.loss, self.train_op], feed_dict)    # loss 标量，损失层（CRF）的输出
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)  # logits: project_layer层的输出
                                                                                # logits: BatchSize*MaxLenSentence*num_tags为全连接层的输出（损失层的输入）
            return lengths, logits      # lengths: batchsize

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [BatchSize, MaxLenSentence, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags + [0]])    # start dimension: num_tag+1
        for score, length in zip(logits, lengths):          # Dimension: score:MaxLenSentence*num_tags; length: 句子实际长度,标量
            score = score[:length]                          # score dimension: LenSentence*num_tags
            pad = small * np.ones([length, 1])              # pad dimension: LenSentence*1
            logits = np.concatenate([score, pad], axis=1)     # logits dimension:LenSentence*(num_tags+1)
            logits = np.concatenate([start, logits], axis=0)  # logits dimension:(1+LenSentence)*(num_tags+1)
            path, _ = viterbi_decode(logits, matrix)        # path dimension: 1+LenSentence

            paths.append(path[1:])          # paths dimension: BatchSize*每句实际长度  每个值是每个字符由模型算出的对应标签ID
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():   # batch dimension: 4*BatchSize*MaxLenSentence
            strings = batch[0]      # string dimension: BatchSize*MaxLenSentence
            tags = batch[-1]        # tags dimension: BatchSize*MaxLenSentence
            lengths, scores = self.run_step(sess, False, batch) # score就是self.logits的计算结果 BatchSize*MaxLenSentence*num_tags
            batch_paths = self.decode(scores, lengths, trans)   # batch_paths dimension: BatchSize*每句实际长度
                                                                # 每个值为每个字符由模型算出的对应标签ID
            for i in range(len(strings)):           # 从batch中取出一句
                result = []
                string = strings[i][:lengths[i]]                                            # string dimension: 第i句实际长度
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])         # gold dimension: 第i句实际长度
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])  # pred dimension: 第i句实际长度
                for char, gold, pred in zip(string, gold, pred):        # 循环一次一个句子
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results      # dimension: eval/test样本数量*每句实际长度  每个三中第一个为字符，第二个为正确标签，第三个为预测标签

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)