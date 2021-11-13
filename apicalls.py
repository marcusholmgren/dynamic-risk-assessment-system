import json
import os
import requests

# Load config.json and get input and output paths
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

output_model_path = os.path.join(config['output_model_path'])

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
response1 = requests.post(URL + "/prediction", json={"data_file": "testdata/testdata.csv"}).text
response2 = requests.get(URL + "/scoring").text
response3 = requests.get(URL + "/summarystats").text
response4 = requests.get(URL + "/diagnostics").text

# combine all API responses
responses = [response1, response2, response3, response4]

# write the responses to your workspace
with open(os.path.join(output_model_path, "apireturns.txt"), 'w') as output_file:
    output_file.writelines(responses)
