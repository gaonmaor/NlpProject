from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from evaluate import exact_match_score, f1_score
from datetime import datetime
import os

import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from evaluate import exact_match_score, f1_score
from config import Config

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        opt_fn = tf.train.AdamOptimizer()
    elif opt == "sgd":
        opt_fn = tf.train.GradientDescentOptimizer(learning_rate=1)  # TODO: Set learn rate.
    else:
        assert False
    return opt_fn


class Encoder(object):
    def __init__(self, state_size, embedding_size):
        self.vocab_dim = embedding_size  # self.vocab_dim = vocab_dim
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.gru_cell = tf.nn.rnn_cell.GRUCell(state_size)
        self.h_q = None
        self.config = None

    def encode(self, question_embeddings, q_mask):  # encode q
        gru_cell = self.gru_cell
        with tf.variable_scope('question_embedding', reuse=tf.AUTO_REUSE):
            seq_length = tf.reduce_sum(tf.cast(q_mask, tf.int32), axis=1)
            # print(len(seq_length))
            outputs, h_q = tf.nn.dynamic_rnn(gru_cell, question_embeddings,
                                             sequence_length=seq_length, dtype=tf.float64)
            # outputs are all the states [batch_size,max_time,state_size] when outputs[:,last,:]=h_q
            output = h_q
        return output  # h_q is the last state from the list list_h_q -- A representationb of Q

    def set_conf(self, conf):
        self.config = conf


class Decoder(object):
    def __init__(self, state_size, embedding_size):
        self.vocab_dim = embedding_size  # self.vocab_dim = vocab_dim
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.gru_cell = tf.nn.rnn_cell.GRUCell(state_size)
        self.outputs = None
        self.config = None
        self.scores_1 = None
        self.scores_2 = None

    def set_conf(self, conf):
        self.config = conf

    def decode(self, paragraph_embeddings, p_mask, h_q):
        gru_cell = self.gru_cell
        with tf.variable_scope('paragraph_embedding', reuse=tf.AUTO_REUSE):
            seq_length = tf.reduce_sum(tf.cast(p_mask, tf.int32), axis=1)
            outputs, h_p = tf.nn.dynamic_rnn(gru_cell, paragraph_embeddings,
                                             sequence_length=seq_length,
                                             dtype=tf.float64, initial_state=h_q)
            W_p_1 = tf.get_variable(shape=(self.config.hidden_size, self.config.max_length_p),
                                    initializer=self.initializer, name='W_p_1', dtype=tf.float64)

            b_p_1 = tf.get_variable(shape=(self.config.max_length_p,), initializer=self.initializer, name='b_p_1',
                                    dtype=tf.float64)  # maybe chanbge to zeros
            scores_1 = tf.matmul(h_p, W_p_1) + b_p_1

            W_p_2 = tf.get_variable(shape=(self.config.hidden_size, self.config.max_length_p),
                                    initializer=self.initializer, name='W_p_2', dtype=tf.float64)
            b_p_2 = tf.get_variable(shape=(self.config.max_length_p,), initializer=self.initializer, name='b_p_2',
                                    dtype=tf.float64)  # maybe chanbge to zeros
            scores_2 = tf.matmul(h_p, W_p_2) + b_p_2
            self.scores_1, self.scores_2 = scores_1, scores_2

        return scores_1, scores_2


class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.config = Config()
        self.scores_1 = None
        self.scores_2 = None
        self.pretrained_embeddings = None
        encoder.set_conf(self.config)
        decoder.set_conf(self.config)

        config = self.config
        # ==== set up placeholder tokens ========
        self.input_placeholder_q = tf.placeholder(tf.int32, shape=(None, config.max_length_q),
                                                  name='q')  # word index in paragraph
        self.mask_placeholder_q = tf.placeholder(tf.bool, shape=(None, config.max_length_q), name='mask_q')

        self.input_placeholder_p = tf.placeholder(tf.int32, shape=(None, config.max_length_p),
                                                  name='p')  # word index in question
        self.mask_placeholder_p = tf.placeholder(tf.bool, shape=(None, config.max_length_p), name='mask_p')

        self.labels_placeholder_p_1 = tf.placeholder(tf.int32, shape=(None,), name='labels_1')
        self.labels_placeholder_p_2 = tf.placeholder(tf.int32, shape=(None,), name='labels_2')

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
        # ==== set up training/updating procedure ==== I think we should call a function here - see q2_rnn for examples

    def pad_sequences(self, data):  # data is int the compact form of word indices
        max_length_q = self.config.max_length_q
        max_length_p = self.config.max_length_p
        """
        max_length_q=100
        max_length_p=600
        """
        zero_vector = 0
        # zero_label = 2 # corresponds to the 'Nop' tag
        ret_p = []
        ret_q = []
        ret_labels = []
        for (par, q, labels) in data:
            len_q = len(q)
            bol_arr_q = np.ones((max_length_q,), dtype=bool)
            if max_length_q > len_q:
                bol_arr_q[len_q:] = False
            q = q[:max_length_q]
            q += (zero_vector for i in range(max_length_q - len_q))
            ret_q.append((q, list(bol_arr_q)))

            len_par = len(par)
            bol_arr = np.ones((max_length_p,), dtype=bool)
            if max_length_p > len_par:
                bol_arr[len_par:] = False
            par = par[:max_length_p]
            # labels=labels[:max_length_p]
            par += (zero_vector for i in range(max_length_p - len_par))
            # labels+=(zero_label for i in range(max_length-len_sen))
            ret_p.append((par, list(bol_arr)))
            ret_labels.append(labels)
        # (data_q,mask_q)=ret_q
        # (data_p,labels,mask_p)
        # print(len(ret_labels))
        return np.array(ret_q), np.array(ret_p), np.array(ret_labels)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        decoder = self.decoder
        encoder = self.encoder

        h_q = encoder.encode(self.embeddings_q, self.mask_placeholder_q)  # encode q
        scores_1, scores_2 = decoder.decode(self.embeddings_p, self.mask_placeholder_p, h_q)
        self.scores_1 = scores_1
        self.scores_2 = scores_2

    def setup_loss(self):
        """
        Set up your loss computation here
        """
        with vs.variable_scope("loss", reuse=tf.AUTO_REUSE):
            my_labels_1 = self.labels_placeholder_p_1
            my_labels_2 = self.labels_placeholder_p_2
            # print(my_labels_1.shape)
            scores_1 = self.scores_1
            scores_2 = self.scores_2
            # print(scores_1.shape)
            # print(scores_1.shape)
            # print(self.labels_placeholder_p_1.shape)

            # produce softmax on max_length
            loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_1, labels=my_labels_1))
            loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_2, labels=my_labels_2))
            self.loss = loss1 + loss2
            self.opt = get_optimizer('adam').minimize(self.loss)

        return self.loss

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """

        self.pretrained_embeddings = np.load(self.config.embed_path)['glove']

        with vs.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            my_size_q = (-1, self.config.max_length_q, self.config.embed_size)  # same size
            my_size_p = (-1, self.config.max_length_p, self.config.embed_size)
            voc = tf.Variable(self.pretrained_embeddings, name='V')  # should make sure it exists
            embeddings_q = tf.nn.embedding_lookup(voc,
                                                  self.input_placeholder_q)  # max_q,embeding - put 0 on the rest#
            embeddings_p = tf.nn.embedding_lookup(voc,
                                                  self.input_placeholder_p)  # max_p,embeding - put zero on the rest #
            self.embeddings_q = tf.reshape(embeddings_q, my_size_q)  # concetenates the embedings embeddings_q
            self.embeddings_p = tf.reshape(embeddings_p, my_size_p)

    def optimize(self, session, par, question, labels, loss):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        p = [x[0] for x in par]
        p_mask = [x[1] for x in par]
        q = [x[0] for x in question]
        q_mask = [x[1] for x in question]

        input_feed = self.create_feed_dict(q, q_mask, p, p_mask, labels)
        output_feed = [self.opt, self.loss]
        _, outputs = session.run(output_feed, input_feed)  # just needs the loss

        return outputs

    def create_feed_dict(self, q, q_mask, p, p_mask, labels):  # train=(q,p), train_y=(labels_e,labels_d)
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        feed_dict = {self.input_placeholder_q: q, self.input_placeholder_p: p, self.mask_placeholder_q: q_mask,
                     self.mask_placeholder_p: p_mask}

        if labels is not None:
            feed_dict[self.labels_placeholder_p_1] = labels[:, 0]
            feed_dict[self.labels_placeholder_p_2] = labels[:, 1]

        return feed_dict

    def decode(self, session, question, par, labels):  # later
        """
        :return: the probability distribution over different
          positions in the paragraph - in my method we just have
          the correct sequance with high a probability of 1 and the rest are 0
        so that other methods like self.answer() will be able to work properly
        """
        # basically 1 at a time
        q, q_mask = question
        p, p_mask = par

        labels = labels.reshape((-1, 2))
        q = q.reshape((-1, self.config.max_length_q))
        q_mask = q_mask.reshape((-1, self.config.max_length_q))
        p = p.reshape((-1, self.config.max_length_p))
        p_mask = p_mask.reshape((-1, self.config.max_length_p))

        # self.encoder.batch_size=labels.shape[0]
        # self.decoder.batch_size=labels.shape[0]

        input_feed = self.create_feed_dict(q, q_mask, p, p_mask, labels)

        output_feed = [self.scores_1, self.scores_2]  # output of the session

        outputs = session.run(output_feed, input_feed)  # [y_e,y_d] return the 2 labelings

        return outputs

    def answer(self, session, question, par, labels):

        yp, yp2 = self.decode(session, question, par, labels)

        a_s = np.argmax(yp, axis=1)[0]
        a_e = np.argmax(yp2, axis=1)[0]

        return a_s, a_e

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should \ the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        # self.encoder.config.batch_size=self.config.batch_size
        # self.decoder.config.batch_size=self.config.batch_size

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        saver = tf.train.Saver()
        """
        results_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        model_path = results_path + "model.weights/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        """
        (ret_q, ret_p, ret_labels) = dataset
        sample_range = np.arange(len(ret_labels))
        iters = int(len(ret_labels) / self.config.batch_size)
        epochs = self.config.n_epochs
        print('epochs: ', epochs, ' iters per epoch: ', iters)
        for epoch in range(epochs):
            tic = time.time()
            print('Epoch: ', epoch)
            for it in range(iters):
                np.random.shuffle(sample_range)
                batch_indices = sample_range[:self.config.batch_size]
                q = ret_q[batch_indices]
                p = ret_p[batch_indices]
                l = ret_labels[batch_indices]
                loss = self.optimize(session, p, q, l, self.loss)
                if it % 100 == 0:
                    print('iter: ', it)
                    print('loss: ', loss)
            toc = time.time()
            logging.info("Complete %d epocs in %d mintes)" % (epoch, int((toc - tic) / 60)))

            # does this works - check where we are loading it
        saver.save(session, "train/model.weights")

    def evaluate_answer(self, session, dataset, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        f1 = 0.
        em = 0.
        (question, par, labels) = dataset

        num_sample = len(labels)
        # why not batch
        for index in range(0, num_sample):
            a_s, a_e = self.answer(session, question[index], par[index], labels[index])
            # print(a_s,a_e)
            answers = par[index][0][a_s: a_e + 1]
            p_s, p_e = labels[index]
            # print(p_s,p_e)
            true_answer = par[index][0][p_s: p_e + 1]

            answers = " ".join(str(a) for a in answers)
            true_answer = " ".join(str(ta) for ta in true_answer)

            # print(answers)
            # print(true_answer)
            f1 += f1_score(answers, true_answer)
            # print('@@@@@@@@@@@@@@@@@@@')
            em += exact_match_score(' '.join(str(a) for a in answers), ' '.join(str(ta) for ta in true_answer))
            # logging.info("answers %s, true_answer %s" % (answers, true_answer))
        f1 /= num_sample
        em /= num_sample

        if log:
            logging.info("F1: {:.2%}, EM: {:.2%}, for {} samples".format(f1, em, num_sample))

        return f1, em
