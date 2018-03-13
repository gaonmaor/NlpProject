"""
Official evaluation script for v1.1 of the SQuAD dataset.
"""
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles
      and extra whitespace.
    :param s: The sentence to normalize.
    :return: The normalized text.
    """
    def remove_articles(text):
        """
        Replace articles with whitespaces.
        :param text: The text to modify.
        :return: The modified text.
        """
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        """
        Replace all whitespaces variations
          with a single space.
        :param text: The text to modify.
        :return: The modified text.
        """
        return ' '.join(text.split())

    def remove_punc(text):
        """
        Remove all punctuations from text.
        :param text: The text to modify.
        :return: The modified text.
        """
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        """
        Lower text characters.
        :param text: The text to modify.
        :return: The modified text.
        """
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Calculates F1 score between predictions and ground_truth
    :param prediction: The predicted answer.
    :param ground_truth: The correct answer.
    :return: The F1 score.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Compare the normalized predictions and ground_truth.
    :param prediction: The predicted answer.
    :param ground_truth: The correct answer.
    :return: True if the prediction is perfect.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Get the max prediction score based on the given metric function.
    :param metric_fn: A function to evaluate the score of
                        the current prediction.
    :param prediction: The predicted answers.
    :param ground_truths: The correct answers.
    :return: The maximum obtained score.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(ds, pred):
    """
    Get exact match and F1 score over all the paragraphs.
    :param ds: Dataset of articles.
    :param pred: Predictions.
    :return: A string of the result.
    """
    f1 = exact_match = total = 0
    for article in ds:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in pred:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = pred[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
