import json
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ingestion')

# Load config.json and get input and output paths
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():
    """
    Merge multiple dataframes into one dataframe
    :return: None
    """
    # check for datasets, compile them together, and write to an output file

    # get all input_files in the input folder
    logger.info("Config key 'input_folder_path': %s", input_folder_path)
    input_files = os.listdir(input_folder_path)
    logger.info("input_files: {}".format(input_files))

    # create a dataframe from each file
    df_list = [pd.read_csv(os.path.join(input_folder_path, file)) for file in input_files]
    raw_data_df = pd.concat(df_list)
    logger.info("Raw dataframe shape: %s", raw_data_df.shape)

    data_df = raw_data_df.drop_duplicates(keep='first')
    logger.info("Duplicates dropped dataframe shape: %s", data_df.shape)

    output_folder_exists = os.path.exists(output_folder_path)
    logger.info("Config output_folder_exists? %s - path: %s", output_folder_exists, output_folder_path)
    if not output_folder_exists:
        os.makedirs(output_folder_path)

    # write the dataframe to an output file
    output_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    logger.info("Output file path: %s", output_file_path)
    data_df.to_csv(output_file_path, index=False)

    # write the input file names to an output file
    output_file_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(output_file_path, 'w') as f:
        ingested_files = [os.path.join(input_folder_path, file) for file in input_files]
        f.write(','.join(ingested_files))


if __name__ == '__main__':
    merge_multiple_dataframe()
