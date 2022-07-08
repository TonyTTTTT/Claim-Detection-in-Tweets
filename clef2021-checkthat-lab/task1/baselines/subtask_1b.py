import glob
import pdb
import pandas as pd
import random
import numpy as np
import logging
import argparse
from os.path import join, dirname, basename

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import sys
sys.path.append('.')

from scorer.subtask_1b import evaluate
from format_checker.subtask_1b import check_format

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))
_COL_NAMES = ['line_number', 'speaker', 'text', 'label']
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


# topic_id tweet_id score run_id
def run_random_baseline(train_debates, results_fpath_og):
    for gold_fpath in train_debates:
        results_fpath = f'{results_fpath_og}_{basename(gold_fpath)}'
        gold_df = pd.read_csv(gold_fpath, names=_COL_NAMES, sep='\t')
        with open(results_fpath, "w") as results_file:
            for i, line in gold_df.iterrows():
                results_file.write('{}\t{}\n'.format(line['line_number'], random.random()))


def run_ngram_baseline(train_debates, test_debates, results_fpath_og):

    train_list = []
    for train_debate in train_debates:
        df = pd.read_csv(train_debate, index_col=None, header=None, names=_COL_NAMES, sep='\t')
        train_list.append(df)
    train_df = pd.concat(train_list)

    test_list = []
    for train_debate in test_debates:
        df = pd.read_csv(train_debate, index_col=None, header=None, names=_COL_NAMES, sep='\t')
        test_list.append(df)
    test_df = pd.concat(test_list)

    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(ngram_range=(1, 1))),
        ('clf', SVC(C=1, gamma=0.75, kernel='rbf', random_state=0))
    ])
    pipeline.fit(train_df['text'], train_df['label'])
    for test_debate in test_debates:
        test_df = pd.read_csv(test_debate, names=_COL_NAMES, sep='\t')
        results_fpath = f'{results_fpath_og}_{basename(test_debate)}'
        with open(results_fpath, "w") as results_file:
            predicted_distance = pipeline.decision_function(test_df['text'])
            for line_num, dist in zip(test_df['line_number'], predicted_distance):
                results_file.write("{}\t{}\n".format(line_num, dist))


def run_baselines(train_debates, dev_debates, lang, subtask='1b'):
    random_baseline_fpath_og = join(ROOT_DIR, f'baselines/data/subtask_{subtask}_random_baseline_{lang}')
    run_random_baseline(dev_debates, random_baseline_fpath_og)
    avg_precisions = []
    for test_debate in dev_debates:
        random_baseline_fpath = f'{random_baseline_fpath_og}_{basename(test_debate)}'
        if check_format(random_baseline_fpath):
            thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate(test_debate, random_baseline_fpath)
            avg_precisions.append(avg_precision)
    logging.info(f"Random Baseline for Subtask-{subtask}--{lang} AVGP: {np.mean(avg_precisions)}")

    ngram_baseline_fpath_og = join(ROOT_DIR, f'baselines/data/subtask_{subtask}_ngram_baseline_{lang}')
    run_ngram_baseline(train_debates, dev_debates, ngram_baseline_fpath_og)
    avg_precisions = []
    for test_debate in dev_debates:
        ngram_baseline_fpath = f'{ngram_baseline_fpath_og}_{basename(test_debate)}'
        if check_format(ngram_baseline_fpath):
            thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate(test_debate, ngram_baseline_fpath)
            avg_precisions.append(avg_precision)
    logging.info(f"Ngram Baseline for Subtask-{subtask}--{lang} AVGP: {np.mean(avg_precisions)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-files-path", "-t", required=True, type=str, nargs='+',
                        help="Path to files to train the baseline models on.")
    
    parser.add_argument("--dev-files-path", "-d", required=True, type=str, nargs='+',
                        help="Path to files to test the basline models on.")
    parser.add_argument("--lang", "-l", default='english', type=str,
                        choices=['english'],
                        help="The language of the subtask")
    args = parser.parse_args()
    run_baselines(args.train_files_path, args.dev_files_path, args.lang, subtask='1b')
