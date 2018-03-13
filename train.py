from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import random
from qa_model import Encoder, QASystem, Decoder, Config
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)


def initialize_model(session, model, train_dir):
    print(train_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements()
                                            for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            lines = f.readlines()
            rev_vocab.extend(lines)
        rev_vocab = [line.decode().strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def load_squad(data_path, prefix, max_vocab, data_dir, max_samples=0):
    prefix_path = pjoin(data_dir, prefix)
    print(prefix_path)

    c_path = prefix_path + ".ids.context"
    if not tf.gfile.Exists(c_path):
        raise ValueError("Context file %s not found.", c_path)

    q_path = prefix_path + ".ids.question"
    if not tf.gfile.Exists(q_path):
        raise ValueError("Question file %s not found.", q_path)

    s_path = prefix_path + ".span"
    if not tf.gfile.Exists(s_path):
        raise ValueError("Span file %s not found.", s_path)

    tic = time.time()
    logging.info("Loading SQUAD data from %s" % prefix_path)

    c_file = open(c_path, mode="rb")
    q_file = open(q_path, mode="rb")
    s_file = open(s_path, mode="rb")

    valid_range = range(0, max_vocab)

    data = []

    max_c = 0
    max_q = 0

    c_buckets = [0] * 10
    q_buckets = [0] * 10

    line = 0
    counter = 0
    for c, q, s in tqdm(zip(c_file, q_file, s_file)):
        line += 1

        c_ids = list(map(int, c.lstrip().rstrip().decode().split(" ")))
        q_ids = list(map(int, q.lstrip().rstrip().decode().split(" ")))
        span = list(map(int, s.lstrip().rstrip().decode().split(" ")))

        if not (len(span) == 2 and span[0] <= span[1] < len(c_ids)):
            # print( "Invalid span at line {}. {} <= {} < {}".format(line, span[0], span[1], len(c_ids)))
            continue

        if max_vocab and not (all(id in valid_range for id in c_ids) and all(id in valid_range for id in q_ids)):
            print("Vocab id is out of bound")
            continue

        data.append((c_ids, q_ids, [span[0], span[1]]))

        c_buckets[len(c_ids) // 100] += 1
        q_buckets[len(q_ids) // 10] += 1

        max_c = max(max_c, len(c_ids))
        max_q = max(max_q, len(q_ids))

        if max_samples and len(data) >= max_samples:
            break

    samples = len(data)

    assert sum(c_buckets) == samples
    assert sum(q_buckets) == samples

    # Sort by context len then by question len
    data.sort(key=lambda tup: len(tup[0]) * 100 + len(tup[1]))

    toc = time.time()
    logging.info("Complete: %d samples loaded in %f secs)" % (samples, toc - tic))
    logging.info("Question length histogram (10 in each bucket): %s" % str(c_buckets))
    logging.info("Context length histogram (100 in each bucket): %s" % str(q_buckets))
    logging.info("Median context length: %d" % len(data[counter // 2][0]))

    return data


def print_sample(sample, rev_vocab):
    print("Context:")
    print(" ".join([rev_vocab[s] for s in sample[0]]))
    print("Question:")
    print(" ".join([rev_vocab[s] for s in sample[1]]))
    print("Answer:")
    print(" ".join([rev_vocab[s] for s in sample[0][sample[2][0]:sample[2][1] + 1]]))


def print_samples(data, n, rev_vocab):
    all_samples = range(len(data))
    for ix in random.sample(all_samples, n) if n > 0 else all_samples:
        print_sample(data[ix], rev_vocab)


def main(_):
    config = Config()
    dataset = None  # TODO ;load dateset ??? - look at dataset and seenhow it loooks - change model.py accordingly

    embed_path = config.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(config.embed_size))
    embeddingz = np.load(embed_path)
    embeddings = embeddingz['glove']
    embeddingz.close()

    vocab_len = embeddings.shape[0]

    train = load_squad(config.data_dir, "train", vocab_len, config.data_dir, max_samples=config.max_train_samples)
    val = load_squad(config.data_dir, "val", vocab_len, config.data_dir, max_samples=config.max_val_samples)

    print('train size: ', len(train), ' val size: ', len(val))

    vocab_path = config.vocab_path or pjoin(config.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    # print_samples(train,1, rev_vocab) #train is triplets of (context,question,answer)
    encoder = Encoder(state_size=config.hidden_size, embedding_size=config.embed_size)
    decoder = Decoder(state_size=config.hidden_size, embedding_size=config.embed_size)

    qa = QASystem(encoder, decoder)
    with tf.Session() as sess:
        load_train_dir = (config.load_train_dir or config.train_dir)  # put "" here if you want to build a new model
        initialize_model(sess, qa, load_train_dir)
        save_train_dir = config.train_dir
        ds_train = qa.pad_sequences(train)
        ret_q, ret_p, ret_labels = ds_train
        qa.train(sess, ds_train, save_train_dir)
        ds_val = qa.pad_sequences(val)

        print('train error')
        qa.evaluate_answer(sess, ds_train, log=True)

        print('val error')
        qa.evaluate_answer(sess, ds_val, log=True)


if __name__ == "__main__":
    tf.app.run()
