from io import StringIO

from typing import List
import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    return #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    return #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time() -> List[float]:
    """
    Calculate the execution time of the model
    :return: List of execution times - ingestion, training
    """
    ingestion_time = timeit.Timer(lambda: subprocess.run(['python', 'ingestion.py'],
                                                         stdout=subprocess.PIPE)).timeit(number=1)
    training_time = timeit.Timer(lambda: subprocess.run(['python', 'training.py'],
                                                        stdout=subprocess.PIPE)).timeit(number=1)
    return [ingestion_time, training_time]


##################Function to check dependencies
def outdated_packages_list():
    """
    Check if any of the dependencies are outdated
    :return:
    """
    logger.info('Checking for outdated packages')
    process = subprocess.run(['pip3', 'list', '--outdated'], stdout=subprocess.PIPE)
    raw_text = process.stdout.decode('utf-8')
    df = pd.read_csv(StringIO(raw_text), index_col='Package', sep=r"\s+", skiprows=[1], engine='python')

    requirements_df = pd.read_csv('requirements.txt', index_col='Package', sep='==',
                                  names=['Package', 'Requirements.txt'], engine='python')

    logger.info("Outdated Python packages\n %s", df.join(requirements_df, on='Package'))


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    timing = execution_time()
    logger.info('Execution time ingestion: %.2f sec, training: %.2f sec', timing[0], timing[1])
    outdated_packages_list()
