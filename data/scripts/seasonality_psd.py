import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import numpy as np
import os

# Configuration
data_folder = 'sliced_data'
file_names = [
    'slice_1.csv', 'slice_2.csv', 'slice_3.csv', 
    'slice_4.csv', 'slice_5.csv', 'slice_6.csv'
]

# Load the datasets
dfs = [pd.read_csv(os.path.join(data_folder, f)) for f in file_names]
df = pd.concat(dfs)

# Convert date to datetime and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.set_index('date')

# Calculate Periodogram
f, Pxx_den = periodogram(df['cases'])

# Plotting
plt.figure(figsize=(5, 4))

# Plot the main spectral density
plt.semilogy(f, Pxx_den, color='blue', alpha=0.7, label='Densité spectrale de puissance')

# Highlight the Weekly frequency (1/7)
weekly_freq = 1/7
# Find index closest to the frequency
idx = (np.abs(f - weekly_freq)).argmin()
#plt.scatter(f[idx], Pxx_den[idx], color='crimson', s=100, zorder=5, label='Composant hebdomadaire (1/7 jours)')

# Highlight harmonics (2/7, 3/7)
harmonics = [2/7, 3/7]
#for harm in harmonics:
   # h_idx = (np.abs(f - harm)).argmin()
   # plt.scatter(f[h_idx], Pxx_den[h_idx], color='orange', s=50, zorder=5, label='Harmonics' if harm == harmonics[0] else "")

# Add vertical line for Weekly
plt.axvline(x=weekly_freq, color='crimson', linestyle='--', alpha=0.8)
plt.text(weekly_freq + 0.01, Pxx_den[idx], 'Cycle hebdomadaire\n(7 jours)', 
         color='crimson', va='center', fontweight='bold')

# Add vertical lines for Harmonics
for harm in harmonics:
    plt.axvline(x=harm, color='orange', linestyle='--', alpha=0.6)

plt.title('Périodogramme (Densité spectrale de puissance)')
plt.xlabel('Frequence(1/jour)')
plt.ylabel('DSP (V**2/Hz)')
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

# Add a top axis to show the Period in Days
ax1 = plt.gca()
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())

# Set ticks at specific frequencies corresponding to days
plt.rcParams.update({'font.size': 16})
tick_locs = [1/7, 2/7, 3/7]
tick_labels = ['7 jours', '3.5 jours', '2.3 jours']
ax2.set_xticks(tick_locs)
ax2.set_xticklabels(tick_labels)
ax2.set_xlabel('Période')

plt.tight_layout()
plt.savefig('periodogram_highlighted.png')