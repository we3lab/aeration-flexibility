#!/usr/bin/env python3
"""
Script to plot daily profiles for up to a week and save them as PNG files.
Replicates the functionality of plot_test_profile.ipynb but saves to files.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import os
from datetime import datetime, timedelta
from pathlib import Path

# Helper functions for standardized plotting
def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 8

def create_subplot_grid(n_rows=2, n_cols=1, figsize=(15, 8)):
    """Create a standardized subplot grid."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    return fig, axes

def standardize_axis(ax, xlabel="Time Step (15 min intervals)", title="", legend_loc='upper right'):
    """Apply standard formatting to an axis."""
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if ax.get_legend():
        ax.legend(bbox_to_anchor=(1.05, 1), loc=legend_loc)
    ax.legend()

def plot_power_profiles(df, date, design_key, ax=None, param_values=None):
    """Plot standardized power profiles."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.plot(df['Edot_b'], label='Edot_b (Blower Power)', color='green', linewidth=1.5)
    ax.plot(df['Edot_t_net'], label='Edot_t_net (Net Power)', color='black', linewidth=1.5)
    ax.plot(df['Edot_c'], label='Edot_c (Charging Power)', color='magenta', linewidth=1.5)
    ax.plot(df['Edot_rem'], label='Edot_rem (Remaining Power)', color='gray', linewidth=1.5)
    ax.plot(df['Edot_t_baseline'], label='Edot_t_baseline (Baseline Power)', color='blue', linewidth=2, linestyle='--')
    
    # Add parameter bounds if available
    if param_values is not None:
        if 'Edot_t_max' in param_values:
            ax.axhline(y=param_values['Edot_t_max'], color='red', linestyle='-', alpha=0.7, 
                      label=f'Edot_t_max: {param_values["Edot_t_max"]:.0f}')
        if 'Edot_b_max' in param_values:
            ax.axhline(y=param_values['Edot_b_max'], color='red', linestyle=':', alpha=0.7, 
                      label=f'Edot_b_max: {param_values["Edot_b_max"]:.0f}')
        if 'Edot_c_max' in param_values:
            ax.axhline(y=param_values['Edot_c_max'], color='red', linestyle='--', alpha=0.7, 
                      label=f'Edot_c_max: {param_values["Edot_c_max"]:.0f}')
    
    ax.set_ylabel('Power (kW)')
    standardize_axis(ax, title=f'Power Profiles for {date} - {design_key}')
    return ax

def plot_flow_profiles(df, date, design_key, ax=None, param_values=None):
    """Plot standardized flow rate profiles with parameter bounds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.plot(df['Ndot_target'], label='Ndot_target (Target Flow)', color='black', linewidth=2, linestyle='--')
    ax.plot(df['Ndot_b'], label='Ndot_b (Blower Flow)', color='red', linewidth=1.5)
    
    # Try to plot gas tank specific flows
    try:
        if 'Ndot_c' in df.columns:
            ax.plot(df['Ndot_c'], label='Ndot_c (Charging Flow)', color='magenta', linewidth=1.5)
        if 'Ndot_r' in df.columns:
            ax.plot(df['Ndot_r'], label='Ndot_r (Recovery Flow)', color='purple', linewidth=1.5)
    except:
        print("Note: No gas tank flows to plot")
    
    ax.plot(df['unmet_o2'], label='Unmet O2 (penalty term)', color='cyan', linewidth=2.0)
    
    # Add parameter bounds if available
    if param_values is not None:
        if 'Ndot_b_min' in param_values:
            ax.axhline(y=param_values['Ndot_b_min'], color='red', linestyle=':', alpha=0.7, 
                      label=f'Ndot_b_min: {param_values["Ndot_b_min"]:.0f} mol/hr')
        if 'Ndot_b_max' in param_values:
            ax.axhline(y=param_values['Ndot_b_max'], color='red', linestyle=':', alpha=0.7, 
                      label=f'Ndot_b_max: {param_values["Ndot_b_max"]:.0f} mol/hr')
    
    ax.set_ylabel('Flow Rate (mol/hr)')
    standardize_axis(ax, title=f'Flow Rate Profiles for {date} - {design_key}')
    return ax

def plot_storage_profiles(df, date, design_key, ax1=None, ax2=None, param_values=None):
    """Plot standardized storage profiles with parameter bounds."""
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    # Storage profiles
    if 'N' in df.columns:
        ax1.plot(df['N'], label='N (O2 Storage)', color='red', linewidth=2)
        ax1.set_ylabel('O2 Storage (mol)')
        ax1.set_title(f'O2 Storage Profile for {date} - {design_key}')
        
        # Add N_min and N_max bounds if available
        if param_values is not None:
            if 'N_min' in param_values:
                ax1.axhline(y=param_values['N_min'], color='red', linestyle='--', alpha=0.7, 
                           label=f'N_min: {param_values["N_min"]:.0f} mol')
            if 'N_max' in param_values:
                ax1.axhline(y=param_values['N_max'], color='red', linestyle='--', alpha=0.7, 
                           label=f'N_max: {param_values["N_max"]:.0f} mol')
        
    elif 'E' in df.columns:
        ax1.plot(df['E'], label='E (Energy Storage)', color='blue', linewidth=2)
        ax1.set_ylabel('Energy Storage (kWh)')
        ax1.set_title(f'Energy Storage Profile for {date} - {design_key}')
        
        # Add E_max bound if available
        if param_values is not None and 'E_max' in param_values:
            ax1.axhline(y=param_values['E_max'], color='blue', linestyle='--', alpha=0.7, 
                       label=f'E_max: {param_values["E_max"]:.0f} kWh')
    
    standardize_axis(ax1)
    
    # Pressure profile (for gas tanks)
    if 'P' in df.columns:
        ax2.plot(df['P'], label='P (Tank Pressure)', color='purple', linewidth=2)
        ax2.set_ylabel('Pressure (MPa)')
        ax2.set_title(f'Tank Pressure Profile for {date} - {design_key}')
        
        if param_values is not None:
            if 'P_min' in param_values:
                ax2.axhline(y=param_values['P_min'], color='purple', linestyle='--', alpha=0.7, 
                            label=f'P_min: {param_values["P_min"]:.3f} MPa')
            if 'P_max' in param_values:
                ax2.axhline(y=param_values['P_max'], color='purple', linestyle='--', alpha=0.7, 
                            label=f'P_max: {param_values["P_max"]:.3f} MPa')
        
        standardize_axis(ax2)
    else:
        ax2.text(0.5, 0.5, 'No pressure data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('No Pressure Data')
        standardize_axis(ax2)

def plot_power_comparison(df, date, design_key, ax1=None, ax2=None):
    """Plot standardized power comparison."""
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Direct comparison
    ax1.plot(df['Edot_t_baseline'], label='Edot_t_baseline', color='blue', linewidth=2)
    ax1.plot(df['Edot_t'], label='Edot_t (Optimized)', color='red', linewidth=2)
    ax1.plot(df['Edot_t_net'], label='Edot_t_net (Net)', color='black', linewidth=1.5, linestyle='--')
    
    ax1.set_ylabel('Power (kW)')
    standardize_axis(ax1, title=f'Power Comparison: Baseline vs Optimized for {date} - {design_key}')
    
    # Difference plot
    power_diff = df['Edot_t'] - df['Edot_t_baseline']
    ax2.plot(power_diff, label='Power Difference (Optimized - Baseline)', color='green', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_ylabel('Power Difference (kW)')
    standardize_axis(ax2, title=f'Power Difference for {date} - {design_key}')
    
    return ax1, ax2

def load_daily_data(run_name, config_name, design_key, date, month_key):
    """Load data for a specific day."""
    intermediate_file = os.path.join(
        f"aeration_flexibility/output_data/{run_name}/intermediate/{config_name}", 
        f"{design_key}_{month_key}_{date}.pkl"
    )
    
    if not os.path.exists(intermediate_file):
        print(f"Not found: {intermediate_file}")
        return None, None, None
    
    with open(intermediate_file, 'rb') as f:
        stored_data = pkl.load(f)
    
    profile = stored_data["profile"]
    new_param_vals = stored_data["new_param_vals"]
    max_values = stored_data["max_values"]
    
    # Convert profile to DataFrame
    processed_data = {}
    for key, value in profile.items():
        if isinstance(value, pd.Series):
            processed_data[key] = value.values
        elif isinstance(value, (list, tuple)):
            processed_data[key] = np.array(value)
        else:
            processed_data[key] = np.array([value] * 96)  # Repeat scalar values for all time steps

    df = pd.DataFrame(processed_data)
    
    return df, new_param_vals, max_values

def generate_date_range(start_date, max_days=7):
    """Generate a list of dates starting from start_date for up to max_days."""
    dates = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    for i in range(max_days):
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return dates

def plot_weekly_profiles(run_name, config_name, design_key, start_date, output_dir="weekly_plots", max_days=7):
    """Plot concatenated profiles for up to a week of data as a single continuous time series."""
    
    # Setup
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate date range
    dates = generate_date_range(start_date, max_days)
    
    # Extract month key from start date
    month_key = start_date[:7]  # "2022-07"
    
    print(f"Plotting profiles for {run_name}/{config_name}/{design_key}")
    print(f"Date range: {start_date} to {dates[-1]}")
    
    # Load and concatenate all available data
    combined_data = []
    param_values = None
    available_dates = []
    
    for date in dates:        
        df, daily_param_values, max_values = load_daily_data(run_name, config_name, design_key, date, month_key)
        
        if df is None:
            continue
        
        # Store param_values from first successful day
        if param_values is None:
            param_values = daily_param_values
        
        # Add day identifier to track where each day starts/ends
        df['day'] = date
        df['day_index'] = len(available_dates)
        
        combined_data.append(df)
        available_dates.append(date)
    
    if not combined_data:
        return
    
    # Concatenate all data into one continuous time series
    full_df = pd.concat(combined_data, ignore_index=True)    
    time_steps = np.arange(len(full_df)) * 0.25  # Convert to hours
    create_combined_weekly_plot(full_df, time_steps, available_dates, design_key, param_values, output_dir, config_name)

def create_combined_weekly_plot(full_df, time_steps, available_dates, design_key, param_values, output_dir, config_name):
    """Create a single combined plot with all data as a continuous time series."""
    
    # Create the main figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Define subplot layout
    ax1 = plt.subplot(3, 1, 1)  # Power profiles
    ax2 = plt.subplot(3, 1, 2)  # Flow profiles  
    ax3 = plt.subplot(3, 1, 3)  # Storage profiles
    
    # Plot 1: Power Profiles
    ax1.plot(time_steps, full_df['Edot_b'], label='Edot_b (Blower Power)', color='green', linewidth=1.5)
    ax1.plot(time_steps, full_df['Edot_t_net'], label='Edot_t_net (Net Power)', color='black', linewidth=1.5)
    ax1.plot(time_steps, full_df['Edot_c'], label='Edot_c (Charging Power)', color='magenta', linewidth=1.5)
    ax1.plot(time_steps, full_df['Edot_rem'], label='Edot_rem (Remaining Power)', color='gray', linewidth=1.5)
    ax1.plot(time_steps, full_df['Edot_t_baseline'], label='Edot_t_baseline (Baseline Power)', color='blue', linewidth=2, linestyle='--')
    
    # Add parameter bounds for power if available
    if param_values is not None:
        if 'Edot_t_max' in param_values:
            ax1.axhline(y=param_values['Edot_t_max'], color='red', linestyle='-', alpha=0.7, 
                      label=f'Edot_t_max: {param_values["Edot_t_max"]:.0f}')
        if 'Edot_b_max' in param_values:
            ax1.axhline(y=param_values['Edot_b_max'], color='red', linestyle=':', alpha=0.7, 
                      label=f'Edot_b_max: {param_values["Edot_b_max"]:.0f}')
    
    ax1.set_ylabel('Power (kW)', fontsize=12)
    ax1.set_title(f'Power Profiles - {design_key} ({available_dates[0]} to {available_dates[-1]})', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Flow Profiles
    ax2.plot(time_steps, full_df['Ndot_target'], label='Ndot_target (Target Flow)', color='black', linewidth=2, linestyle='--')
    ax2.plot(time_steps, full_df['Ndot_b'], label='Ndot_b (Blower Flow)', color='red', linewidth=1.5)
    
    # Try to plot gas tank specific flows
    if 'Ndot_c' in full_df.columns:
        ax2.plot(time_steps, full_df['Ndot_c'], label='Ndot_c (Charging Flow)', color='magenta', linewidth=1.5)
    if 'Ndot_r' in full_df.columns:
        ax2.plot(time_steps, full_df['Ndot_r'], label='Ndot_r (Recovery Flow)', color='purple', linewidth=1.5)
    
    ax2.plot(time_steps, full_df['unmet_o2'], label='Unmet O2 (penalty term)', color='cyan', linewidth=2.0)
    
    # Add parameter bounds for flow if available
    if param_values is not None:
        if 'Ndot_b_min' in param_values:
            ax2.axhline(y=param_values['Ndot_b_min'], color='red', linestyle=':', alpha=0.7, 
                      label=f'Ndot_b_min: {param_values["Ndot_b_min"]:.0f} mol/hr')
        if 'Ndot_b_max' in param_values:
            ax2.axhline(y=param_values['Ndot_b_max'], color='red', linestyle=':', alpha=0.7, 
                      label=f'Ndot_b_max: {param_values["Ndot_b_max"]:.0f} mol/hr')
    
    ax2.set_ylabel('Flow Rate (mol/hr)', fontsize=12)
    ax2.set_title(f'Flow Rate Profiles - {design_key}', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Storage Profiles
    if 'N' in full_df.columns:
        ax3.plot(time_steps, full_df['N'], label='N (O2 Storage)', color='red', linewidth=2)
        ax3.set_ylabel('O2 Storage (mol)', fontsize=12)
        ax3.set_title(f'O2 Storage Profile - {design_key}', fontsize=14)
        
        # Add storage bounds if available
        if param_values is not None:
            if 'N_min' in param_values:
                ax3.axhline(y=param_values['N_min'], color='red', linestyle='--', alpha=0.7, 
                           label=f'N_min: {param_values["N_min"]:.0f} mol')
            if 'N_max' in param_values:
                ax3.axhline(y=param_values['N_max'], color='red', linestyle='--', alpha=0.7, 
                           label=f'N_max: {param_values["N_max"]:.0f} mol')
    
    elif 'E' in full_df.columns:
        ax3.plot(time_steps, full_df['E'], label='E (Energy Storage)', color='blue', linewidth=2)
        ax3.set_ylabel('Energy Storage (kWh)', fontsize=12)
        ax3.set_title(f'Energy Storage Profile - {design_key}', fontsize=14)
        
        # Add energy storage bounds if available
        if param_values is not None and 'E_max' in param_values:
            ax3.axhline(y=param_values['E_max'], color='blue', linestyle='--', alpha=0.7, 
                       label=f'E_max: {param_values["E_max"]:.0f} kWh')
    
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Add vertical lines to show day boundaries
    for i, date in enumerate(available_dates[1:], 1):  # Skip first day
        day_start_hour = i * 24  # Each day is 24 hours
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=day_start_hour, color='gray', linestyle=':', alpha=0.5)
    
    # Set x-axis to show daily marks
    max_hours = len(available_dates) * 24
    day_ticks = np.arange(0, max_hours + 1, 24)
    day_labels = []
    for i, tick in enumerate(day_ticks):
        if i < len(available_dates):
            day_labels.append(f"{available_dates[i]}\n{tick:.0f}h")
        else:
            day_labels.append(f"{tick:.0f}h")
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(day_ticks)
        if ax == ax3:  # Only show labels on bottom plot
            ax.set_xticklabels(day_labels, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    
    # Save the combined plot
    duration_str = f"{len(available_dates)}day" if len(available_dates) > 1 else "1day"
    filename = f"{config_name}_{design_key}_{available_dates[0]}_to_{available_dates[-1]}_{duration_str}_combined.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def create_weekly_summary_plot(run_name, config_name, design_key, dates, month_key, output_dir):
    """Create a summary plot with all days overlaid."""
    
    fig, (ax1, ax2) = create_subplot_grid(2, 1, figsize=(15, 10))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    plotted_days = 0
    
    for i, date in enumerate(dates):
        df, _, _ = load_daily_data(run_name, config_name, design_key, date, month_key)
        
        if df is None:
            continue
        
        color = colors[i % len(colors)]
        
        # Plot power comparison
        ax1.plot(df['Edot_t_baseline'], label=f'{date} (Baseline)', 
                color=color, linewidth=1.5, alpha=0.7)
        ax1.plot(df['Edot_t'], label=f'{date} (Optimized)', 
                color=color, linewidth=2, linestyle='--')
        
        # Plot storage
        if 'N' in df.columns:
            ax2.plot(df['N'], label=f'{date} (O2 Storage)', 
                    color=color, linewidth=2)
        elif 'E' in df.columns:
            ax2.plot(df['E'], label=f'{date} (Energy Storage)', 
                    color=color, linewidth=2)
        
        plotted_days += 1
    
    if plotted_days > 0:
        ax1.set_ylabel('Power (kW)')
        standardize_axis(ax1, title=f'Weekly Power Profiles Comparison - {design_key}')
        
        if 'N' in df.columns:
            ax2.set_ylabel('O2 Storage (mol)')
            standardize_axis(ax2, title=f'Weekly O2 Storage Profiles - {design_key}')
        elif 'E' in df.columns:
            ax2.set_ylabel('Energy Storage (kWh)')
            standardize_axis(ax2, title=f'Weekly Energy Storage Profiles - {design_key}')
        
        plt.tight_layout()
        
        filename = f"{config_name}_{design_key}_weekly_summary.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved weekly summary: {filepath}")

def main():
    """Main function with command line argument parsing."""
    
    # Configuration parameters - modify these to plot different configs and days
    run_name = "wwtp_comparison_svcw_40mgd"  # The run name from your config
    config_name = "air__compressor__1.0__0___compressor__gas_tank__0___0.0__svcw___1.0__0"  # Config name
    design_key = "1.0__100.0"  # Design key (Hours_of_O2__compression_ratio)
    date = "2022-07-05"  # Specific date (YYYY-MM-DD)

    run_name = "nr_comparison_svcw"  # The run name from your config
    config_name = "o2__psa__1.6__0___none__gas_tank__0___0.0__svcw___1.6__0"  # Config name
    design_key = "0.5__700.0"  # Design key (Hours_of_O2__compression_ratio)
    date = "2022-07-01"  # Specific date (YYYY-MM-DD)

    # run_name = "wwtp_comparison_ebmud_40mgd"  # The run name from your config
    # config_name = "air__compressor__1.0__0___compressor__gas_tank__0___0.0__ebmud___1.0__0"  # Config name
    # design_key = "1.0__100.0"  # Design key (Hours_of_O2__compression_ratio)
    # date = "2024-01-03"  # Specific date (YYYY-MM-DD)

    # run_name = "tornado_svcw_psa"  # The run name from your config
    # config_name = "o2__psa__1.1__0___none__gas_tank__0___0.0__svcw_d1.0_e1.0_w0___1.1__0"  # Config name
    # design_key = "1.0__100.0"  # Design key (Hours_of_O2__compression_ratio)
    # date = "2022-07-02"  # Specific date (YYYY-MM-DD)

    # run_name = "speed_test"  # The run name from your config
    # config_name = "air__compressor__1.0__0___compressor__gas_tank__0___0.0__svcw___1.0__0"  # Config name
    # design_key = "3.0__700.0"  # Design key (Hours_of_O2__compression_ratio)
    # date = "2022-07-01"  # Specific date (YYYY-MM-DD)

    # run_name = "speed_test_o2"  # The run name from your config
    # config_name = "air__compressor__1.0__0___compressor__gas_tank__0___0.0__ebmud___1.0__0"  # Config name
    # design_key = "3.0__700.0"  # Design key (Hours_of_O2__compression_ratio)
    # date = "2024-01-01"  # Specific date (YYYY-MM-DD)

    # run_name = "speed_test_o2"  # The run name from your config
    # config_name = "o2__cryo__1.0__0___none__liquid_tank__0___0.0__ebmud___1.0__0"  # Config name
    # design_key = "3.0__100.0"  # Design key (Hours_of_O2__compression_ratio)
    # date = "2024-01-03"  # Specific date (YYYY-MM-DD)
        
    # Run the plotting
    plot_weekly_profiles(
        run_name=run_name,
        config_name=config_name,
        design_key=design_key,
        start_date=date,
        output_dir="weekly_plots",
        max_days=7
    )

if __name__ == "__main__":
    main()
