from io import StringIO

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
def execution_time():
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

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
    execution_time()
    outdated_packages_list()





    
