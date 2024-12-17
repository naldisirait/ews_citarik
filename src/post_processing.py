import numpy as np
import json
import torch
from datetime import datetime, timedelta
import json
import numpy as np
import torch

def is_jsonable(x):
    """
    Check if the input is JSON serializable.
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def convert_to_jsonable(x):
    """
    Convert the input variable to a JSON-compatible type if possible.
    """
    if isinstance(x, (np.ndarray, torch.Tensor)):
        # Convert numpy arrays or torch tensors to lists
        return x.tolist()
    elif isinstance(x, dict):
        # Recursively convert dictionary values
        return {key: convert_to_jsonable(value) for key, value in x.items()}
    elif isinstance(x, list):
        # Recursively convert list elements
        return [convert_to_jsonable(item) for item in x]
    elif isinstance(x, tuple):
        # Convert tuples to lists (tuples are not JSON serializable)
        return [convert_to_jsonable(item) for item in x]
    elif isinstance(x, set):
        # Convert sets to lists
        return list(x)
    else:
        # If it's a basic type or already JSON-serializable, return as is
        return x

def ensure_jsonable(data):
    """
    Ensure the entire data structure is JSON-serializable.
    """
    if is_jsonable(data):
        return data
    else:
        return convert_to_jsonable(data)

def generate_next_24_hours(start_date_str):
    """
    Generate the next 24 hourly timestamps as strings starting from the given date string.
    
    Args:
        start_date_str (str): The starting date as a string.
    
    Returns:
        list: A list of strings representing the next 24 hours.
    """
    # Convert the start_date_str to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')

    # Generate the next 24 hours
    next_24_hours = [(start_date + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(1, 25)]
    
    return next_24_hours

def output_ml1_to_dict(dates, output_ml1, precipitation):
    next_24hr = generate_next_24_hours(dates[-1])
    dates = dates + next_24hr
    time_data = dates[-len(output_ml1):]
    
    # Ensure `precipitation` is serialized
    dict_output_ml1 = {
        "name": "wl", 
        "measurement_type": "forecast",
        "flood event": "yes" if max(output_ml1) > 200 else "no",
        "debit dates": time_data,
        "precipitation value": precipitation.tolist() if isinstance(precipitation, (np.ndarray, torch.Tensor)) else precipitation,
        "debit value": output_ml1  # Ensure this is a list, already handled by output_ml1.tolist() before
    }
    return dates, dict_output_ml1

def output_ml2_to_dict(dates, output_ml2):
    output_ml2[output_ml2 < 0.1] = 0
    
    # Convert output_ml2 to a list
    dict_output_ml2 = {
        "name": "max_depth",
        "start_date": dates[0],
        "end_date": dates[-1],
        "inundation": output_ml2.tolist()  # This ensures it's serialized
    }
    return dict_output_ml2

def convert_array_to_tif(data_array, filename, meta=None):
    """
    Function to convert 2D array into .tif data
    Args:
        data_array: 2D array, float value
        meta: meta of the tif as georeference for creating the .tif
        filename: output filename.tif
    Return: None, this function will not return any value
    """
    #import modul rasterio and affine
    from affine import Affine
    import rasterio
    from rasterio.crs import CRS

    #check if meta is provided or not
    if not meta:
        meta ={'driver': 'GTiff',
               'dtype': 'float32',
               'nodata': -9999.0,
               'width': 1680,
               'height': 1621,
               'count': 1,
               'crs': CRS.from_epsg(32750),
               'transform': Affine(5.0, 0.0, 815899.0,0.0, -5.0, 9902502.0968)}
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(data_array, 1)
