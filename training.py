import argparse
import json
import logging
import os
import pickle

import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info(', '.join(df.columns))
    one_hot = pd.get_dummies(df['corporation'])
    df = df.drop(['corporation'], axis=1)
    df = pd.concat([df, one_hot], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df.drop([args.target], axis=1), df[args.target], test_size=0.2)

    model.fit(X_train, y_train)
    logger.info('Model score %.4f', model.score(X_test, y_test))

    logger.info('Model accuracy_score %.4f', metrics.accuracy_score(y_test, model.predict(X_test)))
    logger.info('Model roc_auc_score %.4f', metrics.roc_auc_score(y_test, model.predict(X_test)))
    logger.info('Model precision_score %.4f', metrics.precision_score(y_test, model.predict(X_test)))
    logger.info('Model recall_score %.4f', metrics.recall_score(y_test, model.predict(X_test)))
    logger.info('Model f1_score %.4f', metrics.f1_score(y_test, model.predict(X_test)))

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
