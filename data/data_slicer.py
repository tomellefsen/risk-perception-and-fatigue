import pandas as pd
import os
from config import SLICES

# -----------------------------------
# CONFIGS

SLICE_OUTPUT_FOLDER = 'sliced_data'
os.makedirs(SLICE_OUTPUT_FOLDER, exist_ok=True)

try:
    df = pd.read_csv('cases.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').set_index('date')
    
    # Calculate the 7-day rolling average here too, so it's included in the output files
    # df['cases_smooth'] = df['cases'].rolling(window=7, center=True).mean()

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print(f"Saving {len(SLICES)} data slices to the '{SLICE_OUTPUT_FOLDER}' folder...")

slice_number = 0
for slice_data in SLICES:
    start_date = slice_data['start']
    end_date = slice_data['end']
    slice_number = slice_number + 1
    
    # Slice the DataFrame
    df_slice = df.loc[start_date:end_date]
    
    # Create a clean filename
    filename = os.path.join(SLICE_OUTPUT_FOLDER, f'slice_{slice_number}.csv')
    
    # Save to CSV
    df_slice.to_csv(filename)
    print(f"-> Saved: {filename}")

print("\nData slicing complete.")