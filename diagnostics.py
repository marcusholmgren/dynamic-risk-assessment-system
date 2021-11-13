import json
import logging
import os
import pickle
import subprocess
import timeit
from io import StringIO
from typing import List, Tuple

import pandas
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('diagnostics')


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
production_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path']) 


def model_predictions(df: pandas.DataFrame) -> List[float]:
    """
    Get model predictions for the test data
    :param df: pandas dataframe
    :return: List of predictions from dataframe
    """
    # read deployed model
    with open(os.path.join(production_deployment_path, 'trainedmodel.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)

    # encode corporation code to numeric value
    df['corporation'] = df['corporation'].apply(lambda x: sum(bytearray(x, 'utf-8')))
    X = df.drop(['exited'], axis=1)

    y_pred = model.predict(X)
    logger.info('Length of data: %s vs. predictions: %s', df.shape[0], len(y_pred))
    return y_pred.tolist()


def dataframe_summary() -> List[Tuple[str, List[float]]]:
    """
    Get summary statistics of the dataframe
    :return: List of columns summary statistics of mean, median and std
    """
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    summary = df.select_dtypes('number').agg(['mean', 'median', 'std'])
    sum_stats = [(col, summary[col].values.tolist()) for col in summary.columns]
    return sum_stats


def dataframe_missing_values() -> List[str]:
    """
    Get summary statistics of the dataframe missing values
    :return: List of columns missing values percentage
    """
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    na_stats = df.isna().sum() / df.count().sum()
    stats_report = [f"{x[0]} pct NA: {x[1]}" for x in zip(na_stats.index, na_stats.tolist())]
    return stats_report


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


def outdated_packages_list():
    """
    Check if any of the dependencies are outdated
    :return:
    """
    logger.info('Checking for outdated packages')
    process = subprocess.run(['pip3', 'list', '--outdated'], stdout=subprocess.PIPE)
    raw_text = process.stdout.decode('utf-8')
    if raw_text:
        logger.info('Outdated packages: %s', raw_text)
        df = pd.read_csv(StringIO(raw_text), index_col='Package', sep=r"\s+", skiprows=[1], engine='python')

        requirements_df = pd.read_csv('requirements.txt', index_col='Package', sep='==',
                                      names=['Package', 'Requirements.txt'], engine='python')

        logger.info("Outdated Python packages\n %s", df.join(requirements_df, on='Package'))
    else:
        logger.info('Yay! all packages are up to date')


if __name__ == '__main__':
    with open(os.path.join(test_data_path, 'testdata.csv'), 'r') as test_data:
        test_data_df = pd.read_csv(test_data)

    preds = model_predictions(test_data_df)
    logger.info('Predictions: %s', preds)
    stats = dataframe_summary()
    logger.info('Summary statistics: %s', stats)
    na_stats = dataframe_missing_values()
    logger.info("Missing values stats: %s", na_stats)
    timing = execution_time()
    logger.info('Execution time ingestion: %.2f sec, training: %.2f sec', timing[0], timing[1])
    outdated_packages_list()
