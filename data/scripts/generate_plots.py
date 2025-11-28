"""
Generates and saves a suite of COVID-19 case plots.

This module loads the cleaned ``cases.csv`` file and the ``SLICES``
configuration from ``config.py``. It calculates a 7-day rolling average
on the case data.

It defines four distinct plotting functions:
1.  ``generate_overview_plot``: A single plot of the entire timeline with
    all key policy events from all slices annotated.
2.  ``generate_sliced_plot``: Individual plots for each specific time slice.
3.  ``generate_combined_slices_plot``: A single plot of the entire timeline
    with colored spans highlighting each slice.
4.  ``generate_grid_plot``: A 3x2 grid figure containing all individual
    slice plots for easy comparison.

When executed as a script, it calls all plotting functions and saves
the resulting ``.png`` files to the ``PLOT_OUTPUT_FOLDER``.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

from config import SLICES

# -----------------------------------
# CONFIGS & LOADING

PLOT_OUTPUT_FOLDER = 'covid_plots'
os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)

try:
    df = pd.read_csv('cases.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').set_index('date')
    df['cases_smooth'] = df['cases'].rolling(window=7, center=True).mean()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()
    
# -----------------------------------
# PLOTTING FUNCTIONS

# 1. Overview Plot (Full Timeline)
def generate_overview_plot(df_full, slices_data, output_folder):
    """
    Generates a plot of the full timeline with all events annotated.

    This function aggregates all unique events from all provided slices
    and plots them on a single graph of the entire smoothed case timeline.
    Event labels are staggered vertically to improve readability.

    Parameters
    ----------
    df_full : pd.DataFrame
        The complete, time-indexed DataFrame containing 'cases_smooth'.
    slices_data : list of dict
        The list of slice definitions (from ``config.py``). Each dict
        must contain an 'events' key.
    output_folder : str
        The directory path where the plot image will be saved.

    Returns
    -------
    None
        Saves the plot as 'annotated_overview_plot_fixed.png' in the
        ``output_folder``.
    """
    plt.figure(figsize=(18, 8))
    
    y_max = df_full['cases_smooth'].max()
    y_lim = y_max * 1.25
    y_stagger_increment = y_max * 0.05 

    all_events_overview = []
    for slice_data in slices_data:
        all_events_overview.extend(slice_data['events'])
    unique_events = sorted(list(set(all_events_overview)), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))

    stagger_count = 0 
    previous_date = None
    
    plt.plot(df_full.index, df_full['cases_smooth'], color='blue', linewidth=2, label='7-Day Rolling Average')

    for date_str, label, color in unique_events:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        # ... (rest of the annotation logic is unchanged)
        if previous_date and (date - previous_date).days > 3:
            stagger_count = 0
            
        y_pos = y_max + y_stagger_increment * stagger_count

        plt.axvline(x=date, color=color, linestyle='--', linewidth=1, alpha=0.7)
        plt.text(date, y_pos, label, 
                 rotation=90, verticalalignment='top', color=color, fontsize=9, 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
        
        stagger_count += 1
        previous_date = date

    plt.title('Daily COVID-19 Cases in Quebec: Full Timeline and Key Policy Events', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Daily Cases (Smoothed)', fontsize=14)
    ax = plt.gca()
    ax.set_ylim(0, y_lim)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    filename = os.path.join(output_folder, 'annotated_overview_plot_fixed.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# 2. Individual Sliced Plots
def generate_sliced_plot(df_full, start_date, end_date, slice_name, events, output_folder):
    """
    Generates a plot for a single, specific time slice.

    This function filters the main DataFrame to the given date range
    and plots both the raw 'cases' and the 'cases_smooth' data. It
    annotates the plot with the specific events provided for this slice.

    Parameters
    ----------
    df_full : pd.DataFrame
        The complete, time-indexed DataFrame containing 'cases' and
        'cases_smooth'.
    start_date : str or pd.Timestamp
        The start date for the slice (inclusive).
    end_date : str or pd.Timestamp
        The end date for the slice (inclusive).
    slice_name : str
        A descriptive name for the slice, used in the plot title
        and filename.
    events : list of tuple
        A list of events specific to this slice. Each tuple should be
        in the format (date_str, label, color).
    output_folder : str
        The directory path where the plot image will be saved.

    Returns
    -------
    None
        Saves the plot as 'slice_{slice_name}_fixed.png' in the
        ``output_folder``.
    """
    df_slice = df_full.loc[start_date:end_date].copy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_slice.index, df_slice['cases'], color='lightgray', linewidth=1)
    
    if 'cases_smooth' not in df_slice.columns:
        df_slice['cases_smooth'] = df_slice['cases'].rolling(window=7, center=True).mean()
        
    plt.plot(df_slice.index, df_slice['cases_smooth'], color='firebrick', linewidth=2, label='7-Day Rolling Average')

    y_max = df_slice['cases_smooth'].max()
    y_lim = y_max * 1.40 
    y_stagger_increment = y_max * 0.07 

    if not df_slice.empty:
        stagger_count = 0
        previous_date = None
        
        for date_str, label, color in events:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            # ... (rest of the annotation logic is unchanged)
            if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
                
                if previous_date and (date - previous_date).days > 3:
                    stagger_count = 0

                y_pos = y_max + y_stagger_increment * stagger_count
                
                plt.axvline(x=date, color=color, linestyle='--', linewidth=1, alpha=0.7)
                plt.text(date, y_pos, label, 
                         rotation=90, verticalalignment='top', color=color, fontsize=10, 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
                
                stagger_count += 1
                previous_date = date

    plt.title(f'COVID-19 Cases in Quebec: {slice_name}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Cases (Smoothed)', fontsize=12)
    ax = plt.gca()
    ax.set_ylim(0, y_lim)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    filename = os.path.join(output_folder, f'slice_{slice_name.replace(" ", "_")}_fixed.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# 3. Combined Slices Plot
def generate_combined_slices_plot(df_full, slices_data, output_folder):
    """
    Generates a plot of the full timeline with colored spans for each slice.

    This function plots the entire smoothed case timeline and then uses
    ``axvspan`` to draw semi-transparent colored rectangles over the
    background, corresponding to the start and end dates of each slice.
    It adds a legend to identify the slices.

    Parameters
    ----------
    df_full : pd.DataFrame
        The complete, time-indexed DataFrame containing 'cases_smooth'.
    slices_data : list of dict
        The list of slice definitions (from ``config.py``). Each dict
        must contain 'start', 'end', and 'name' keys.
    output_folder : str
        The directory path where the plot image will be saved.

    Returns
    -------
    None
        Saves the plot as 'combined_slices_plot_fixed.png' in the
        ``output_folder``.
    """
    plt.figure(figsize=(18, 8))
    
    plt.plot(df_full.index, df_full['cases_smooth'], color='blue', linewidth=4, label='Moyenne Mobile 7 Jours')

    slice_colors = ["#91CDE2", "#71E671", "#F0F060", "#F27F52", "#DF72DF", "#B6B6B6"] 
    
    y_max = df_full['cases_smooth'].max()
    y_lim = y_max * 1.05
    ax = plt.gca()
    ax.set_ylim(0, y_lim)
    
    for i, slice_data in enumerate(slices_data):
        start_date = pd.to_datetime(slice_data['start'])
        end_date = pd.to_datetime(slice_data['end'])
        color = slice_colors[i % len(slice_colors)]
        
        plt.axvspan(start_date, end_date, color=color, alpha=0.2)
        
        mid_date = start_date + (end_date - start_date) / 2
        text_y_pos = y_max * 0.95
        
        if i > 0:
            plt.axvline(start_date, color='black', linestyle='--', linewidth=1, alpha=0.6)
        
        name = slice_data['name'].replace('_', ' ').replace('Épidémie', 'Ép.').replace('confinement', 'conf.')
        plt.text(mid_date, text_y_pos, name, 
                 rotation=90, verticalalignment='top', horizontalalignment='center', 
                 fontsize=10, color='black', alpha=0.8,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.title('Cas COVID-19 au Québec: Périodes Modélisées', fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Cas Quotidiens (moyenne 7 jours)', fontsize=15)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.6)
    
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.3) for c in slice_colors]
    legend_labels = [s['name'].replace('_', ' ') for s in slices_data]
    line_handle = plt.Line2D([0], [0], color='blue', linewidth=2, label='Moyenne Mobile 7 Jours')
    
    #plt.legend([line_handle] + legend_handles, ['Moyenne Mobile 7 Jours'] + legend_labels, 
    #           loc='upper right', ncol=2, fontsize=12)

    plt.legend("")
    plt.tight_layout()
    
    filename = os.path.join(output_folder, 'combined_slices_plot_fixed.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# 4. Grid Sliced Plots
def generate_grid_plot(df_full, slices_data, output_folder):
    """
    Generates a 3x2 grid of plots, each showing an individual slice.

    This function creates a single figure with 6 subplots (axes) and
    iterates through the ``slices_data``, placing each individual
    slice plot (using the same logic as ``generate_sliced_plot``) onto
    one of the subplots.

    Parameters
    ----------
    df_full : pd.DataFrame
        The complete, time-indexed DataFrame containing 'cases' and
        'cases_smooth'.
    slices_data : list of dict
        The list of slice definitions (from ``config.py``). This
        is expected to have 6 entries to fill the 3x2 grid.
    output_folder : str
        The directory path where the plot image will be saved.

    Returns
    -------
    None
        Saves the plot as 'sliced_plots_3x2_grid.png' in the
        ``output_folder``.
    
    Notes
    -----
    The subplot layout is hardcoded to 3 rows and 2 columns. If the
    number of slices changes from 6, this layout may need adjustment.
    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18), constrained_layout=True)
    axes = axes.flatten()

    for i, slice_data in enumerate(slices_data):
        ax = axes[i]
        
        start_date = slice_data['start']
        end_date = slice_data['end']
        slice_name = slice_data['name'].replace('_', ' ').replace('Épidémie', 'Ép.').replace('confinement', 'conf.')
        events = slice_data['events']
        
        df_slice = df_full.loc[start_date:end_date].copy()
        
        if df_slice.empty:
            ax.set_title(f'{slice_name} (No Data)')
            ax.axis('off')
            continue

        ax.plot(df_slice.index, df_slice['cases'], color='lightgray', linewidth=1)
        
        if 'cases_smooth' not in df_slice.columns:
            df_slice['cases_smooth'] = df_slice['cases'].rolling(window=7, center=True).mean()

        ax.plot(df_slice.index, df_slice['cases_smooth'], color='firebrick', linewidth=2, label='7-Day Rolling Average')

        y_max = df_slice['cases_smooth'].max()
        y_lim = y_max * 1.40
        y_stagger_increment = y_max * 0.07 

        stagger_count = 0
        previous_date = None
        
        for date_str, label, color in events:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
                
                if previous_date and (date - previous_date).days > 3:
                    stagger_count = 0

                y_pos = y_max + y_stagger_increment * stagger_count
                
                ax.axvline(x=date, color=color, linestyle='--', linewidth=1, alpha=0.7)
                ax.text(date, y_pos, label, 
                         rotation=90, verticalalignment='top', color=color, fontsize=9, 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
                
                stagger_count += 1
                previous_date = date

        ax.set_title(f'COVID-19 Cases: {slice_name}', fontsize=20)
        ax.set_ylabel('Daily Cases (Smoothed)', fontsize=14)
        ax.set_ylim(0, y_lim)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        if i >= 4:
            ax.set_xlabel('Date', fontsize=16)
        else:
            ax.tick_params(labelbottom=True)
            
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.6)
        ax.legend(loc='upper left', fontsize=8)
        
    fig.suptitle('COVID-19 Daily Cases in Quebec: Evolution by Policy Period (3x2 Grid)', fontsize=20, y=1.0)
    
    filename = os.path.join(output_folder, 'sliced_plots_3x2_grid.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# -----------------------------------

if __name__ == '__main__':
    print("Starting plot generation...")
    
    # Generate all plots, passing the imported SLICES
    generate_overview_plot(df, SLICES, PLOT_OUTPUT_FOLDER)
    print("1. Overview plot generated.")
    
    for slice_data in SLICES:
        safe_slice_name = slice_data['name'].replace('é', 'e').replace('è', 'e').replace('_', '')
        generate_sliced_plot(df, slice_data['start'], slice_data['end'], safe_slice_name, slice_data['events'], PLOT_OUTPUT_FOLDER)
    print("2. Individual sliced plots generated.")
    
    generate_combined_slices_plot(df, SLICES, PLOT_OUTPUT_FOLDER)
    print("3. Combined slices plot generated.")
    
    generate_grid_plot(df, SLICES, PLOT_OUTPUT_FOLDER)
    print("4. Grid sliced plots generated.")

    print(f"\nAll plots saved successfully to the '{PLOT_OUTPUT_FOLDER}' folder.")