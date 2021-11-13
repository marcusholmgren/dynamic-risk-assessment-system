import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

from diagnostics import model_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('reporting')


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])


def score_model(args: argparse.Namespace):
    """
    Function that calculate confusion matrix and writes it to workspace.
    :param args:
    :return: None
    """
    data_file = 'testdata.csv'
    df = pd.read_csv(os.path.join(test_data_path, data_file))
    logger.info('Loaded dataset %s', df.shape)
    y = df['exited']
    y_preds = model_predictions(df)

    # calculate the confusion matrix
    cm = metrics.confusion_matrix(y, y_preds)
    _ = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix from {data_file}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # write the confusion matrix to the workspace
    plt.savefig(os.path.join(config['output_folder_path'], args.conf_matrix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create reports on ML model")

    parser.add_argument('--conf_matrix',
                        type=str,
                        required=False,
                        default='confusionmatrix.png')
    arguments = parser.parse_args()
    score_model(arguments)
