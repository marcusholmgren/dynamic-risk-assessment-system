import argparse
import json
import logging
import os
import pickle

import pandas as pd
from sklearn import metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('scoring')


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])


def score_model(args: argparse.Namespace):
    """
    Function to score the model
    :param args:
    :return: None
    """
    logger.info('Loading model: %s/%s', model_path, args.model_name)
    with open(os.path.join(model_path, args.model_name), 'rb') as model_file:
        model = pickle.load(model_file)

    logger.info('Loading test data: %s/%s', test_data_path, args.test_data_name)
    test_df = pd.read_csv(os.path.join(test_data_path, args.test_data_name))

    # encode corporation code to numeric value
    test_df['corporation'] = test_df['corporation'].apply(lambda x: sum(bytearray(x, 'utf-8')))
    y = test_df['exited']
    X = test_df.drop(['exited'], axis=1)

    y_preds = model.predict(X)
    f1_score = metrics.f1_score(y, y_preds)
    logger.info('F1 score: %s on test data.', f1_score)

    logger.info('Writing F1 score to %s/%s', model_path, args.model_score)
    with open(os.path.join(model_path, args.model_score), 'w') as score_file:
        score_file.write(str(f1_score))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Score a model')

    args.add_argument('--model_name',
                      type=str,
                      help='Name of model to score',
                      required=False,
                      default='trainedmodel.pkl')

    args.add_argument('--test_data_name',
                      type=str,
                      help='Name of test data file',
                      required=False,
                      default='testdata.csv')

    args.add_argument('--model_score',
                      type=str,
                      help='Name of file to write score to',
                      required=False,
                      default='latestscore.txt')

    arguments = args.parse_args()
    score_model(arguments)
