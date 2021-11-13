
import json
import logging
import os
import subprocess

import pandas as pd
from sklearn import metrics

from diagnostics import model_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fullprocess')

# Load config.json and get input and output paths
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def go():
    new_files = check_for_new_data()
    # if you found new data, you should proceed. otherwise, do end the process here
    if new_files:
        ingest_new_data()
        model_drift = check_for_model_drift()
        if model_drift:
            retrain_model()
            redeploy_model()
            diagnostic_and_reporting()


def check_for_new_data() -> bool:
    """
    Checks that the data files are the sames as the one in the ingested file
    :return: bool True if new data is found, False if not
    """
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'r') as ingested_file:
        ingested_files = set(ingested_file.read().split(','))
    logging.info('Ingested files %s', ingested_files)
    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    input_files = set([os.path.join(input_folder_path, file) for file in os.listdir(input_folder_path)])
    is_same = len(ingested_files) == len(ingested_files.union(input_files))
    difference = input_files.difference(ingested_files)
    logging.info('Ingestion no files are same? %s -- new files: %s', is_same, difference)
    return bool(len(difference))


def ingest_new_data():
    """
    Runs the ingest process
    :return: None
    """
    subprocess.run(['python', 'ingestion.py'], stdout=subprocess.PIPE)


def check_for_model_drift() -> bool:
    """
    Checks whether the model has drifted
    :return:
    """
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as score_file:
        latest_score = float(score_file.read())

    df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    y = df['exited']
    y_pred = model_predictions(df)
    new_score = metrics.f1_score(y, y_pred)

    logging.info('F1 score previous: %.4f vs new: %.4f', latest_score, new_score)
    return latest_score < new_score


def retrain_model():
    """
    Runs the training process
    :return:
    """
    subprocess.run(['python', 'training.py'], stdout=subprocess.PIPE)


def redeploy_model():
    """
    Runs the deployment process
    :return:
    """
    subprocess.run(['python', 'deployment.py'], stdout=subprocess.PIPE)


def diagnostic_and_reporting():
    """
    Runs the scoring, diagnostics and reporting process
    :return:
    """
    subprocess.run(['python', 'scoring.py'], stdout=subprocess.PIPE)
    subprocess.run(['python', 'diagnostics.py'], stdout=subprocess.PIPE)
    subprocess.run(['python', 'reporting.py'], stdout=subprocess.PIPE)


if __name__ == '__main__':
    go()
