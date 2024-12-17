from datetime import timedelta,datetime
import datetime as DateTime
import xarray as xr
import os
from ftplib import FTP
import numpy as np

def slice_data_to_indonesia(xr_data):
    xr_indo = xr_data.sel(Latitude=slice(-15,15), Longitude=slice(89.94999, 149.95))
    return xr_indo

def slice_data_to_palu(xr_data):
    #potong data hanya pada bagian DAS saja
    left, right, top, bottom = 119.741082, 120.332054, -0.837767, -1.589718
    xr_palu = xr_data.sel(Latitude=slice(-1.589718, -0.837767), Longitude=slice(119.741082, 120.332054))
    return xr_palu
    
def convert_time_gsmap_now(unit):
    str_time = unit[12:]
    # Given datetime object
    datetime_obj = datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S.%f")
    
    # Add 1 hour to the datetime object
    datetime_obj_plus_1_hour = datetime_obj + timedelta(hours=1)

    return datetime_obj_plus_1_hour
    
def get_downloaded_filenames(path):
    files = os.listdir(path)
    return files

def get_input_precip_data(path_data, days):
    """
    Get input dataset from gsmap now
    Parameters:
        path_data: string, path to half hourly gsmap
    Returns:
        prec_val: array like, precipitation value for the last 72 days
    """
    number_of_data = days * 24

    #open all the data
    data = xr.open_mfdataset(f"{path_data}/*.nc")

    #sum the to make it hourly precip
    summed_data = data.coarsen(Time=2, boundary='trim').sum()

    #get the variable GC
    prec_val = summed_data['hourlyPrecipRateGC'].values
    
    #get only the latest data needed for prediction
    if len(prec_val) > number_of_data:
        prec_val = prec_val[-number_of_data:,:,:]
        
    #because the data originally hourly, but the units is mm/hr so we need to divide the value by 2
    prec_val = prec_val/2
    
    return prec_val


def get_data_from_ftp(ftp, path_data, filename):
    #set filename raw data
    path_to_save = f"{path_data}/gsmap_now_raw/{filename}"
    
    # Open the local file in write mode
    with open(path_to_save, 'wb') as local_file:
        # Retrieve the file from the FTP server and write it to the local file
        ftp.retrbinary('RETR ' + filename, local_file.write)
        
    # Open nc data
    xr_data = xr.open_dataset(path_to_save,decode_times = False)
    
    #slice data
    xr_data_indo = slice_data_to_indonesia(xr_data)
    xr_data_palu = slice_data_to_palu(xr_data_indo)
    
    #convert date into real date
    real_time = convert_time_gsmap_now(xr_data_indo['Time'].attrs['units'])
    xr_data_indo['Time'] = [real_time]
    xr_data_palu['Time'] = [real_time]
    
    #save dataset
    formatted_real_time = real_time.strftime('%Y_%m_%d_%H_%M')
    xr_data_indo.to_netcdf(f'{path_data}/gsmap_now_indo/gsmap_now_rain_indo_{str(formatted_real_time)}.nc')
    xr_data_palu.to_netcdf(f'{path_data}/gsmap_now_palu/gsmap_now_rain_palu_{str(formatted_real_time)}.nc')
    
def get_data_gsmap_now(path_data):
    # FTP server details
    ftp_server = 'hokusai.eorc.jaxa.jp'
    ftp_username = 'rainmap'
    ftp_password = 'Niskur+1404'
    
    # Connect to the FTP server
    ftp = FTP(ftp_server)
    ftp.login(user=ftp_username, passwd=ftp_password)
    
    # Format current date
    formatted_date = DateTime.datetime.now(DateTime.UTC).strftime("%Y/%m/%d")
    
    # Navigate to the directory containing the file
    ftp.cwd(f"/now/netcdf/{formatted_date}")

    #get downloaded data
    downloaded_file_nc = get_downloaded_filenames(f"{path_data}/gsmap_now_raw")

    #sort data name
    file_list = sorted(ftp.nlst())

    #filter filename, we just want to download the nc. not the flag
    file_nc = [i for i in file_list if 'flag' not in i]

    assert len(file_list) > 0, "The folder is empty"
    
    t_back = 48 #Total data to download
    latest_file = file_nc[-t_back:]
    print(f"Total data avaible in {formatted_date} is {len(file_nc)}")
    
    for filename in latest_file:
        if filename not in downloaded_file_nc:
            print(f"Downloading file {filename}")
            get_data_from_ftp(ftp=ftp,path_data=path_data,filename=filename)
        else:
            print(f"Data {filename} already exists")
        
    # Close the FTP connection
    ftp.quit()