#import module from global env
import numpy as np
import torch
import os
import yaml
import time
from datetime import datetime

#import module from this projects
from src.utils import get_current_datetime
from models.discharge.model_ml1 import inference_ml1
from models.inundation.model_ml2 import inference_ml2
from src.data_ingesting import get_input_ml1, get_input_ml1_hujan_max
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict, ensure_jsonable

# Load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def do_prediction(t0=None):
    
    # Get configuration to run the system
    start_run_time = get_current_datetime()
    if t0:
        t0 = datetime.strptime(t0, '%Y-%m-%d %H:%M:%S')

    config_path = 'config.yaml'
    config = load_config(config_path)
    input_size_ml2 = config['model']['input_size_ml2']

    #1. Ingest data hujan
    t_start_ingest = time.time()

    #write the code here

    t_end_ingest = time.time()

    print(f"Succesfully ingesting the data: {t_end_ingest-t_start_ingest}s")

    #2. Predict debit using ML1
    input_ml1 = torch.rand(1, 72) #this is dummy input
    t_start_ml1 = time.time()
    debit = inference_ml1(input_ml1,config)
    input_ml2 = debit[-input_size_ml2:].view(1,input_size_ml2,1)
    t_end_ml1 = time.time()

    print(f"Succesfully inference ml1: {t_end_ml1-t_start_ml1}s")

    #3. Predict inundation using ML2
    t_start_ml2 = time.time()
    genangan = inference_ml2(input_ml2)
    t_end_ml2 = time.time()
    print(f"Succesfully inference ml2: {t_end_ml2-t_start_ml2}s")
    end_run_time = get_current_datetime()

    # #4. Bundle output
    # dates, dict_output_ml1 = output_ml1_to_dict(dates=date_list, output_ml1=debit.tolist(), precipitation=ch_wilayah)
    # dict_output_ml2 = output_ml2_to_dict(dates=dates[-input_size_ml2:],output_ml2=genangan)

    # output = {"Prediction Time Start": str(start_run_time), 
    #         "Prediction Time Finished": str(end_run_time),
    #         "precip information": data_information,
    #         "precip source":data_name_list,
    #         "precip date": date_list,
    #         "Prediction Output ML1": dict_output_ml1,
    #         "Prediction Output ML2": dict_output_ml2}
    
    # output = ensure_jsonable(output)
    # return output

# Run the application using the following command:
# uvicorn app2:app --reload
