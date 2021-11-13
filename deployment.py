import json
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deployment')


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


def prepare_deployment():
    """
    Copy files needed for deployment.
    :return:
    """
    def copy_file(filename, path):
        try:
            src = os.path.join(path, filename)
            dst = os.path.join(prod_deployment_path, filename)
            if os.path.isfile(src):
                logger.info('Copying file %s to %s', src, dst)
                shutil.copy(src, dst)
            else:
                logger.warning('File %s does not exist', src)
        except:
            logger.error("Error copying file: ", src)

    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)
    copy_file('ingestedfiles.txt', dataset_csv_path)
    copy_file('trainedmodel.pkl', output_model_path)
    copy_file('latestscore.txt', output_model_path)
    logger.info('Files copied to %s', prod_deployment_path)


if __name__ == '__main__':
    prepare_deployment()
