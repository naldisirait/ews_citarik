import numpy as np
import xarray as xr
import pickle
from scipy.interpolate import griddata
import torch
import os
from datetime import datetime, timedelta

def pre_process_data_stasiun_pupr(df):
    # Split 'coordinate' into two new columns 'longitude' and 'latitude'
    df[['longitude', 'latitude']] = df['coordinate'].str.split(',', expand=True)
    
    # Convert the new columns to float type for numerical operations
    df['longitude'] = df['longitude'].astype(float)
    df['latitude'] = df['latitude'].astype(float)

    #list_stasiun = ['CH OMU','CH TONGOA','SAMBO','CH INTAKE LEWARA','CH BANGGA BAWAH','CH TUVA', 'CH INTAKE POBOYA']
    list_stasiun = df['name'].values.tolist()
    data_input = {}
    for stasiun in list_stasiun:
        stas_df = df[df['name'] == stasiun]
        if len(stas_df) > 0:
            lat,lon,prec = stas_df['latitude'].values[0], stas_df['longitude'].values[0], stas_df['rainfall'].values[0]
            data_input[stasiun] = {"prec": prec, "latitude" : lat, "longitude": lon}
    return data_input

def interpolate_station_to_grided_palu(input_station_data):
    station_latitudes, station_longitudes, precipitation_values = [],[],[]
    for stas,val in input_station_data.items():
        station_latitudes.append(val['latitude'])
        station_longitudes.append(val['longitude'])
        precipitation_values.append(val['prec'])
        
    station_latitudes, station_longitudes, precipitation_values = np.array(station_latitudes), np.array(station_longitudes), np.array(precipitation_values)
    # Combine lat, lon into a 2D array of station coordinates
    station_coords = np.column_stack((station_longitudes, station_latitudes))
    # Define the grid over the target area, with latitudes and longitudes
    
    grid_latitudes = np.linspace(-0.55, -1.85, 14)
    grid_longitudes = np.linspace(119.15, 120.75, 17)
    
    # Create meshgrid for the target grid points (latitude, longitude)
    grid_lon, grid_lat = np.meshgrid(grid_longitudes[6:13], grid_latitudes[3:11])
    
    # Interpolate precipitation values onto the grid using IDW (linear)
    grid_precipitation = griddata(station_coords, precipitation_values, (grid_lon, grid_lat), method='nearest')
    return grid_precipitation
    
def get_hdfs_path_gsmap(date):
    date_str = date.strftime('%Y%m%d.%H%M')  # Format the date as 'YYYYMMDD.HHMM'
    hdfs_path = f"hdfs://master-01.bnpb.go.id:8020/user/warehouse/JAXA/curah_hujan/{date.strftime('%Y/%m/%d')}/gsmap_now_rain.{date_str}.nc"
    return hdfs_path
    
def slice_data_to_palu(xr_data):
    #potong data hanya pada bagian DAS Palu saja
    left, right, top, bottom = 119.75, 120.35, -0.85, -1.55
    xr_palu = xr_data.sel(Latitude=slice(bottom, top), Longitude=slice(left, right))
    return xr_palu
    
def get_prec_only_palu(file_path):
    ds = xr.open_dataset(file_path, decode_times=False)
    ds_palu = slice_data_to_palu(ds)
    # Flip the latitude dimension (reverse the order)
    ds_palu = ds_palu.isel(Latitude=slice(None, None, -1))
    prec_values = ds_palu['hourlyPrecipRateGC'][0].values
    return prec_values

def get_grided_prec_palu(date):
    hdfs_path = get_hdfs_path_gsmap(date)
    print(f"Trying to get the precipitation Palu from GSMAP Jaxa {hdfs_path}")
    from pyspark.sql import SparkSession
    filename = hdfs_path[-31:]
    local_path = f'./data/gsmap/{filename}'
    spark = SparkSession.builder \
        .appName("master") \
        .config("spark.hadoop.hadoop.security.authentication", "kerberos") \
        .config("spark.hadoop.hadoop.security.authorization", "true") \
        .config("spark.security.credentials.hive.enabled","false") \
        .config("spark.security.credentials.hbase.enabled","false") \
        .enableHiveSupport().getOrCreate()
    
    # Mengakses FileSystem melalui JVM
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

    # Membuat objek Path di HDFS dan lokal
    hdfs_file_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
    local_file_path = spark._jvm.org.apache.hadoop.fs.Path(local_path)

    # Gunakan FileUtil untuk menyalin dari HDFS ke sistem lokal
    spark._jvm.org.apache.hadoop.fs.FileUtil.copy(fs, hdfs_file_path, spark._jvm.org.apache.hadoop.fs.FileSystem.getLocal(hadoop_conf), local_file_path, False, hadoop_conf)
    
    prec_val_palu = get_prec_only_palu(local_path)
    os.remove(local_path)
    return prec_val_palu
    
def get_precip_pupr(date):
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("master") \
            .config("spark.hadoop.hadoop.security.authentication", "kerberos") \
            .config("spark.hadoop.hadoop.security.authorization", "true") \
            .config("spark.security.credentials.hive.enabled","false") \
            .config("spark.security.credentials.hbase.enabled","false") \
            .enableHiveSupport().getOrCreate()
        date = date.strftime('%Y-%m-%d_%H-%M-%S')
        path = f"hdfs://master-01.bnpb.go.id:8020/user/warehouse/SPLP/PUPR/curah_hujan/palu/curah_hujan_{date}.json"
        json_data = spark.read.option("multiline","true").json(path)
        df = json_data.toPandas()
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Not Available", None
    return "Available", df

def correct_data_gsmap(data,df_correction_factor_path):
    #open correction factors 
    with open(df_correction_factor_path, 'rb') as file:
        df_correction_factor = pickle.load(file)

    # Initialize a zeros array to store corrected values
    corrected_data = np.zeros_like(data)
    # Apply the correction and store results in the corrected_data array
    for idx, row in df_correction_factor.iterrows():
        below_threshold = row['below_threshold']
        upper_threshold = row['upper_threshold']
        correct_bias = row['correct_bias']
        
        # Create a mask where values are within the threshold range
        mask = (data >= below_threshold) & (data <= upper_threshold)
        
        # Store the corrected values in the corrected_data array
        corrected_data[mask] = data[mask] * correct_bias

    print("Returns Corrected Data GSMAP")
    return corrected_data
    
def get_precip_gsmap(date):
    try:
        early_30_min_date = date - timedelta(minutes=30)
        prec_value_1 = get_grided_prec_palu(early_30_min_date)
        prec_value_2 = get_grided_prec_palu(date)
        hourly_prec = (prec_value_1 + prec_value_2) / 2
        return "Available", hourly_prec
    except Exception as e:
        print(f"Error occurred: {e}")
        return ("Not Available", None)
    
def get_data_from_biglake(date,config,correct_data=True):
    # Simulate fetching data from biglake (PUPR or GSMAP)
    pupr_avaibility, df = get_precip_pupr(date)
    print(date, "pupr", pupr_avaibility)
    if pupr_avaibility == "Available":
        data_input = pre_process_data_stasiun_pupr(df)
        grided_data = interpolate_station_to_grided_palu(data_input)
        return "pupr", grided_data
    else:
        date = date - timedelta(hours=8)
        jaxa_avaibility, data = get_precip_gsmap(date)
        if jaxa_avaibility=="Available":
            if correct_data:
                df_correction_factor_path = config['data_processing']['gsmap_correction_factor_path']
                data = correct_data_gsmap(data,df_correction_factor_path)
            return "gsmap", data
        else:
            return None
        
def get_data_precip_72jam(path_hujan_72_jam, config, t0=None):
    # Step 1: Get time prediction start (t0)
    if not t0:
        t0 = datetime.now()
    # Step 2: Convert t0's minutes and seconds to zero
    t0 = t0.replace(minute=0, second=0, microsecond=0)

    # Step 3: Generate 72 hourly dates backward (including t0)
    hourly_dates = [t0 - timedelta(hours=i) for i in range(72)][::-1]

    # Step 4: Create an empty list to store the data
    data_list = []
    data_name_list = []
    date_list = []
    with open(path_hujan_72_jam, 'rb') as file:
        stored_data = pickle.load(file)
        
    # Step 5: Loop through the generated 72 hourly dates
    new_dict = {}
    for date in hourly_dates:
        str_date = str(date)
        date_list.append(str_date)
        #Check if the date exists in hujan_72_jam.pkl
        if (str_date in stored_data) and (stored_data[str_date]['data name'] != "no data"):
            prec_value = stored_data[str_date]['prec']
            data_name = stored_data[str_date]['data name']
            new_dict[str_date] = {'prec':prec_value, 'data name':data_name}
            data_list.append(prec_value)
            data_name_list.append(data_name)
        else:
            # If date doesn't exist on the stored data, fetch from biglake (PUPR or GSMAP)
            try:
                data_name, prec_value = get_data_from_biglake(date,config)
                data_list.append(prec_value)
                data_name_list.append(data_name)
                new_dict[str_date] = {'prec': prec_value, 'data name': data_name}
            except Exception as e:
                print(f"Error occurred while fetching data for date {str_date}: {e}")
                prec_value = np.zeros((8,7))
                data_name = "no data"
                data_name_list.append(data_name)
                data_list.append(prec_value)
                new_dict[str_date] = {'prec': prec_value, 'data name': data_name}
              
    #Update data file 72 jam
    with open(path_hujan_72_jam, 'wb') as file:
        # Step 3: Dump the dictionary into the file
        pickle.dump(new_dict, file)
        
    no_data_list = [i for i in data_name_list if i == "no data"]
    half_data = len(data_name_list)/2
    
    if len(no_data_list)>half_data:
        data_information = "Not Good, 50 percent data missed"
    else:
        data_information = "Good"
    return np.array(data_list), date_list, data_information,data_name_list

def get_input_ml1(t0, config):
    path_hujan_72_jam = config['data_processing']['path_hujan_hist_72jam']
    data_hujan, date_list, data_information, data_name_list, = get_data_precip_72jam(path_hujan_72_jam=path_hujan_72_jam,config=config,t0=t0)
    data_hujan = data_hujan.tolist()
    data_hujan = np.array(data_hujan)
    t,w,h = data_hujan.shape
    ch_wilayah = np.mean(data_hujan, axis = (1,2))
    input_ml1= torch.tensor(data_hujan)
    input_ml1 = input_ml1.view(1,t,w,h)
    return input_ml1, ch_wilayah, date_list, data_information, data_name_list

def get_input_ml1_hujan_max():
    """
    function to get max precipitation data to illustrate largest possible flood event.
    """
    path_hujan_max = "./data/data_hujan_max.pkl"
    # Open the file in binary read mode and load the object
    with open(path_hujan_max, 'rb') as file:
        data = pickle.load(file)
        value = data['data_input_max']

        #get ch wilayah for input max hujan
        value_np = np.array(value)[0]
        ch_wilayah = np.mean(value_np, axis = (1,2))
        ch_wilayah = ch_wilayah.tolist()

        #get tensor value for input ml1
        value = torch.tensor(value, dtype=torch.float)
    return value, ch_wilayah
