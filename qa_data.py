from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import argparse
from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin
import gzip
import arff
import tarfile
from six.moves import urllib

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2


def setup_args():
    """
    Parse the configuration arguments.
    :return: The arguments.
    """
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "squad")
    glove_dir = os.path.join("download", "dwr")
    source_dir = os.path.join("data", "squad")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()


def basic_tokenizer(sentence):
    """
    Tokenize a sentence into single spaced words.
    :param sentence: The sentence to be tokenized.
    :return: A list of words.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        # print(space_separated_fragment)
        space_separated_fragment = "".join(map(chr, space_separated_fragment))
        words.extend(re.split(" ", space_separated_fragment))       
    return [w for w in words if w]


def initialize_vocabulary(vocabulary_path):
    """

    :param vocabulary_path:
    :return:
    """
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    """

    :param args:
    :param vocab_list:
    :param save_path:
    :param size:
    :param random_init:
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        found = 0
        #                line=bytes(line,'utf-8')
        #        line=line.decode('utf-8'

        # with arff.load(open(glove_path)) as fh:
        fh=open(glove_path,'rb')
        
        for line in tqdm(fh, total=size):
            line=line.decode('utf-8')
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in vocab_list:
                idx = vocab_list.index(word)
                glove[idx, :] = vector
                found += 1
            if word.capitalize() in vocab_list:
                idx = vocab_list.index(word.capitalize())
                glove[idx, :] = vector
                found += 1
            if word.upper() in vocab_list:
                idx = vocab_list.index(word.upper())
                glove[idx, :] = vector
                found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(
            found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    """

    :param vocabulary_path:
    :param data_paths:
    :param tokenizer:
    :return:
    """
    print(vocabulary_path)
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="r") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                w=bytes(str(w), 'utf-8')
                vocab_file.write(w + b"\n")


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    """

    :param sentence:
    :param vocabulary:
    :param tokenizer:
    :return:
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None):
    """

    :param data_path:
    :param target_path:
    :param vocabulary_path:
    :param tokenizer:
    :return:
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


if __name__ == '__main__':
    print('create_vocabulary')
    args = setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    train_path = pjoin(args.source_dir, "train")
    valid_path = pjoin(args.source_dir, "val")
    dev_path = pjoin(args.source_dir, "dev")

    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "train.context"),
                       pjoin(args.source_dir, "train.question"),
                       pjoin(args.source_dir, "val.context"),
                       pjoin(args.source_dir, "val.question")])
    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below
    print('process_glove')
    # process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim),
    #              random_init=args.random_init)

    # ======== Creating Dataset =========
    # We created our data files seperately
    # If your model loads data differently (like in bulk)
    # You should change the below code
    print('data_to_token_ids_1')
    x_train_dis_path = train_path + ".ids.context"
    y_train_ids_path = train_path + ".ids.question"
    data_to_token_ids(train_path + ".context", x_train_dis_path, vocab_path)
    data_to_token_ids(train_path + ".question", y_train_ids_path, vocab_path)
    print('data_to_token_ids_2')
    x_dis_path = valid_path + ".ids.context"
    y_ids_path = valid_path + ".ids.question"
    data_to_token_ids(valid_path + ".context", x_dis_path, vocab_path)
    data_to_token_ids(valid_path + ".question", y_ids_path, vocab_path)
