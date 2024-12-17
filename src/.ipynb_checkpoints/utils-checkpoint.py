import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def generate_hourly_dates(start_time_str, n=145, m=71):
    # Parse the start time string to a datetime object
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
    
    # Calculate the starting point in the list of dates
    initial_time = start_time - timedelta(hours=m)
    
    # Generate a list of n hourly dates
    hourly_dates = [initial_time + timedelta(hours=i) for i in range(n)]
    
    # Convert the list of datetime objects to a Pandas DatetimeIndex
    hourly_dates_pandas = pd.to_datetime(hourly_dates)
    
    return hourly_dates_pandas

def visualize_sample_input_output_discharge(X, y, precip=None, save=False, predictions = None, sample_index=None):
    """
    Visualize a sample of the input and target sequences, and optionally a third variable on a secondary y-axis.

    Parameters:
    X (np.ndarray): Array containing the input sequences.
    y (np.ndarray): Array containing the target sequences.
    precip (np.ndarray, optional): Array containing the third variable to be plotted on a secondary y-axis.
    sample_index (int, optional): Index of the sample to visualize. If None, a random sample is selected.
    """
    if sample_index is None:
        sample_index = np.random.randint(0, len(X))
    
    input_seq = X[sample_index]
    target_seq = y[sample_index]
    
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Get current axes
    ax1.plot(range(len(input_seq)), input_seq, label='Input Discharge Sequence')
    ax1.plot(range(len(input_seq), len(input_seq) + len(target_seq)), target_seq, label='Target Discharge Sequence', color='orange')
    ax1.set_title(f'Sample {sample_index} - Discharge Input and Target Sequences')
    ax1.axvline(x=len(input_seq), color='r', linewidth=1, linestyle='--', label='Prediction Starts')


    if predictions is not None:
        predicted_seq = predictions[sample_index]
        ax1.plot(range(len(input_seq), len(input_seq) + len(target_seq)), predicted_seq, label='Predicted Sequence',color = 'blue', linestyle='--')
    
    if precip is not None:
        ax2 = ax1.twinx()  # Create a second y-axis
        ax2.plot(range(len(input_seq)), precip[sample_index], label='Input Precipitation Sequence', color='green', linestyle='--')
        ax2.set_ylabel('Precipitation (mm/hr)')
        ax2.legend(loc='upper right')
    ax1.set_xlabel('Time Steps (hr)')
    ax1.set_ylabel('Discharge Value ($m^3$)')
    ax1.legend(loc='upper left')
    ax1.grid(False)
    if save:
        plt.savefig(f'Sample_{sample_index}.png', bbox_inches ='tight')
    plt.show()

def visualize_debit(data, title, start_prediction):
    arr_data = np.array(data)
    assert len(arr_data.shape) == 1, "Data must be 1 dimensional"
    # Convert date strings to datetime objects
    # dates = pd.date_range(start=start_prediction, periods=len(data), freq='h')
    
    dates = generate_hourly_dates(start_time_str=start_prediction, n=len(data), m=71)

    title = f"Hourly debit start from {dates[0]} to {dates[-1]}"
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, data, marker='o')
    plt.xlabel('Date')
    plt.ylabel('Value (m3/hr)')
    plt.title(title)
    
    # Formatting x-ticks
    plt.xticks(rotation=25, fontsize = 7)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(15))  # Show a maximum of 15 x-ticks
    
    # Format x-ticks to show date and hour
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    