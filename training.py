import argparse
import json
import logging
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('training')

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


def train_model(args: argparse.Namespace):
    """
    Train the model
    :param args: argparse.Namespace
    :return: None
    """

    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    logger.info('Loading data from %s/%s', dataset_csv_path, args.input_artifact)
    df = pd.read_csv(os.path.join(dataset_csv_path, args.input_artifact))
    # encode corporation code to numeric value
    df['corporation'] = df['corporation'].apply(lambda x: sum(bytearray(x, 'utf-8')))

    y = df[args.target]
    X = df.drop([args.target], axis=1)

    model.fit(X, y)
    logger.info('Model score %.4f on training data.', model.score(X, y))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info('Writing trained model to %s/%s', model_path, args.output_model)
    with open(os.path.join(model_path, args.output_model), 'wb') as model_file:
        pickle.dump(model, model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')

    parser.add_argument('--input_artifact',
                        type=str,
                        help='Input artifact name',
                        required=False,
                        default='finaldata.csv')

    parser.add_argument('--target',
                        type=str,
                        help='Target column name',
                        required=False,
                        default='exited')

    parser.add_argument('--output_model',
                        type=str,
                        help='Output model name',
                        required=False,
                        default='trainedmodel.pkl')

    arguments = parser.parse_args()
    train_model(arguments)
