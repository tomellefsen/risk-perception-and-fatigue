"""
Demonstrates the presence of seasonality (weekly harmonic) 
variation in case reporting, justifying the need to 
implement this behavior in the observation model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import os

# Configuration: Define the subfolder and file names
data_folder = 'sliced_data'
file_names = [
    'slice_1.csv', 'slice_2.csv', 'slice_3.csv', 
    'slice_4.csv', 'slice_5.csv', 'slice_6.csv'
]

# Load the datasets from the specified subfolder
# os.path.join handles the directory separators (slashes) automatically
dfs = [pd.read_csv(os.path.join(data_folder, f)) for f in file_names]

# Concatenate all dataframes
df = pd.concat(dfs)

# Convert date to datetime and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.set_index('date')

# Calculate ACF
# We use nlags=25 to clearly show up to lag 21
acf_values = sm.tsa.acf(df['cases'], nlags=25)
lags = np.arange(len(acf_values))

# Calculate significance threshold (95% confidence interval)
# The threshold is approximately +/- 1.96 / sqrt(N)
significance_thresh = 1.96 / np.sqrt(len(df))

# Plotting
plt.figure(figsize=(8, 6))

# Plot all lags as standard bars
plt.bar(lags, acf_values, width=0.3, color='lightgray', label='Autres délais')

# Highlight specific lags
highlight_lags = [7, 14, 21]
plt.bar(highlight_lags, acf_values[highlight_lags], width=0.3, color='crimson', label='Délais hebdomadaire (7, 14, 21)')

# Add lines/stems for better visualization (lollipop chart style)
for lag in lags:
    plt.plot([lag, lag], [0, acf_values[lag]], color='gray', alpha=0.5, linestyle='-')

for lag in highlight_lags:
    plt.plot([lag, lag], [0, acf_values[lag]], color='crimson', linewidth=2)

# Add significance lines
plt.axhline(y=significance_thresh, color='blue', linestyle='--', alpha=0.5, label='Seuil de signification 95%')
plt.axhline(y=-significance_thresh, color='blue', linestyle='--', alpha=0.5)
plt.axhline(y=0, color='black', linewidth=1)

# Annotate the specific values
for lag in highlight_lags:
    plt.text(lag, acf_values[lag] + 0.05, f'{acf_values[lag]:.2f}', 
             ha='center', va='bottom', color='crimson', fontweight='bold')

plt.rcParams.update({'font.size': 16})
plt.title('Fonction d\'autocorrélation (ACF)')
plt.xlabel('Délais (Days)')
plt.ylabel('Coefficient d\'autorégulation')
plt.xticks(lags)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('acf_highlighted.png')