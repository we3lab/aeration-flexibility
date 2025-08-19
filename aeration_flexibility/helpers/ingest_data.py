import numpy as np
import pandas as pd
import pickle
import os
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from helpers.parameters import cb_palette

from helpers.compressor_power import *
from helpers.parameters import *

from flows_prep import prep_data, imputation
from flows_prep.utils import utils as ut
from flows_prep.utils import skeleton as skel
from pype_schema.parse_json import JSONParser


def clean_data_with_flows_prep(ingest_gas, network_filename, parameter_filename, facility_id, source_filenames, 
               date_range, run_name="run_20250707_o2", scale_factor=None):
    """Clean EBMUD OR SVCW data using flows_prep modules"""
    
    aeration_flexibility_dir = Path(__file__).parent.parent
    json_file_path = aeration_flexibility_dir / "data" / "facility" / ingest_gas / network_filename
    parameters_path = aeration_flexibility_dir / "data" / "facility" / ingest_gas / parameter_filename
    output_dir = f"aeration_flexibility/output_data/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    source_paths = []
    for i in range (len(source_filenames)):
        source_paths.append(str(aeration_flexibility_dir / "data" / "facility" / ingest_gas / source_filenames[i]))

    if not json_file_path.exists():
        raise FileNotFoundError(f"Network file not found at {json_file_path}")
    
    # Load the network from JSON
    print(f"parsing {json_file_path}")
    parser = JSONParser(str(json_file_path))
    print(f"parsed")
    network = parser.initialize_network(verbose=False)
    print("initialized")

    # Create or update parameters file
    if not os.path.exists(parameters_path):
        parameters = skel.create_params_skeleton(network)
        print("created skeleton")
        # Update the general info for the facility
        if facility_id in parameters:
            parameters[facility_id]["GENERAL_INFO"].update({
                "facility_id": facility_id,
                "datetime_varname": "DateTime",
                "available_date_range": date_range,
                "clean_date_range": date_range,
                "exclude_dts": [],
                "consec_nan_limit": 0,
                "source_path": source_paths
            })
            
            # Update all source_path fields to use the main source path
            main_source_path = parameters[facility_id]["GENERAL_INFO"]["source_path"]
            for var_name, var_config in parameters[facility_id]["SCADA_VARIABLES"].items():
                if "preprocessing" in var_config:
                    var_config["preprocessing"]["source_path"] = main_source_path
        with open(parameters_path, 'w') as f:
            json.dump(parameters, f, indent=3)
    else:
        # Load existing parameters and update them
        with open(parameters_path, 'r') as f:
            parameters = json.load(f)
        print("loaded skeleton")
        # Update parameters to reflect current network
        parameters = skel.update_parameters(parameters, network, remove_deleted_tags=True)
        print("updated skeleton")
        # Update the general info for the facility to ensure source paths are correct
        if facility_id in parameters:
            parameters[facility_id]["GENERAL_INFO"].update({
                "facility_id": facility_id,
                "datetime_varname": "DateTime",
                "available_date_range": date_range,
                "clean_date_range": date_range,
                "exclude_dts": [],
                "consec_nan_limit": 0,
                "source_path": source_paths
            })
            
            # Update all source_path fields to use the main source path
            main_source_path = parameters[facility_id]["GENERAL_INFO"]["source_path"]
            for var_name, var_config in parameters[facility_id]["SCADA_VARIABLES"].items():
                if "preprocessing" in var_config:
                    var_config["preprocessing"]["source_path"] = main_source_path
        
        # Save updated parameters file
        with open(parameters_path, 'w') as f:
            json.dump(parameters, f, indent=3)
        print("saved updated skeleton")
    
    # Handle source_path as a list, not a string
    general_info = parameters[facility_id]["GENERAL_INFO"]
    if isinstance(general_info["source_path"], str):
        general_info["source_path"] = [general_info["source_path"]]
    print("creating prepper")
    prepper = prep_data.DataPrepper(
        parameters[facility_id], 
        network,
        resolution="15m",
        verbose=False
    )

    # Stage 1: Preprocess raw data
    print("prepping data")
    processed_data = prepper.prep_raw_data(verbose=True)
    
    print("converting to numeric")
    # Convert all columns to numeric, coercing errors to NaN (except DateTime)
    for col in processed_data.columns:
        if col != 'DateTime':  # Skip DateTime column to preserve datetime functionality
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
    
    # Stage 2: Clean processed data
    print("cleaning data")
    cleaned_data = prepper.prep_clean_data(processed_data)

    # Save intermediate data files
    processed_data.to_csv(f"{output_dir}/processed_data.csv", index=False)
    cleaned_data.to_csv(f"{output_dir}/cleaned_data.csv", index=False)

    # Stage 3: Impute missing data
    get_imputed = True
    get_final = False

    if get_imputed:
        imputed_data = prepper.prep_imputed_data(cleaned_data, bootstrap_impute=False)
        imputed_data.to_csv(f"{output_dir}/imputed_data.csv", index=False)
    else:
        imputed_data = cleaned_data
    
    # Stage 4: Final data preparation
    if get_final:
        final_data = prepper.prep_final_data(imputed_data)
        final_data.to_csv(f"{output_dir}/final_data.csv", index=False)
    else:
        final_data = imputed_data
    
    # Stage 5: Scale flow and power if scale_factor is provided
    scaled_data = None
    flow_power_keywords = ['flow', 'power', 'electricity', 'energy', 'blower', 'compressor', 'cogenerator', 'battery']
    exclude_cols = ['DateTime', 'hour', 'month', 'year', 'date', '15_minute_key', 'index']
    
    flow_power_columns = [
        col for col in final_data.columns 
        if any(keyword in col.lower() for keyword in flow_power_keywords) 
        and col not in exclude_cols
    ]
    
    print(f"Scaling {len(flow_power_columns)} flow/power columns: {flow_power_columns}")
    scaled_data = final_data.copy()
    scaled_data[flow_power_columns] *= scale_factor
    scaled_data.to_csv(f"{output_dir}/scaled_data.csv", index=False)

    # Prepare data for plotting by adding time-based columns
    def prepare_data_for_plotting(data):
        if data is None or len(data) == 0:
            return None
        
        data_copy = data.copy()
        if not isinstance(data_copy.index, pd.DatetimeIndex):
            if "DateTime" in data_copy.columns:
                data_copy = data_copy.set_index("DateTime")
            else:
                data_copy.index = pd.to_datetime(data_copy.index)
        data_copy["hour"] = data_copy.index.hour
        data_copy["month"] = data_copy.index.month
        data_copy["year"] = data_copy.index.year
        data_copy["date"] = data_copy.index.date
        
        return data_copy
    
    # Prepare each stage for plotting
    processed_for_plotting = prepare_data_for_plotting(processed_data)
    cleaned_for_plotting = prepare_data_for_plotting(cleaned_data)
    imputed_for_plotting = prepare_data_for_plotting(imputed_data)
    final_for_plotting = prepare_data_for_plotting(final_data)
    scaled_for_plotting = prepare_data_for_plotting(scaled_data)

    # Generate plots for each stage
    plot_data_stages(processed_for_plotting, 
                     cleaned_for_plotting, 
                     imputed_for_plotting, 
                     final_for_plotting, 
                     scaled_for_plotting,
                     run_name, ingest_gas, scale_factor)

    return scaled_data, final_data


def get_export_profiles_standardized(data, ingest_gas):
    """Calculate normalized multipliers and extract raw values with standardized profile names"""
    if ingest_gas == 'air':
        # print(f"  Blower_1_AerationBasin_Air_Flow: {data['Blower_1_AerationBasin_Air_Flow'].mean():.2f}")
        # print(f"  Blower_2_AerationBasin_Air_Flow: {data['Blower_2_AerationBasin_Air_Flow'].mean():.2f} ft³/min")
        # print(f"  Blower_3_AerationBasin_Air_Flow: {data['Blower_3_AerationBasin_Air_Flow'].mean():.2f} ft³/min")
        # print(f"  Blower_4_AerationBasin_Air_Flow: {data['Blower_4_AerationBasin_Air_Flow'].mean():.2f} ft³/min")
        # print(f"  Blower_5_AerationBasin_Air_Flow: {data['Blower_5_AerationBasin_Air_Flow'].mean():.2f} ft³/min")
        # print(f"  Blower_6_AerationBasin_Air_Flow: {data['Blower_6_AerationBasin_Air_Flow'].mean():.2f} ft³/min")
        # print(f"  Blower_7_AerationBasin_Air_Flow: {data['Blower_7_AerationBasin_Air_Flow'].mean():.2f} ft³/min")
        # print(f"  Blower_8_AerationBasin_Air_Flow: {data['Blower_8_AerationBasin_Air_Flow'].mean():.2f}")
        # print(f"  Blower_AerationBasin_Air_Flow: {data['Blower_AerationBasin_Air_Flow'].mean():.2f}")
        # print(f"  Blower_AerationBasin_Total_Air_Flow: {data['Blower_AerationBasin_Total_Air_Flow'].mean():.2f}")
        
        export_profiles = {
            "15 Minute Key": data.index.values,
            "LiftPumpToPrimary": data["LiftPump_PrimaryClarifier_UntreatedSewage_Flow"].values,
            "Blower_AerationBasin_Air_Flow": data["Blower_AerationBasin_Air_Flow"].values,
            "AerationBasin_TotalAerationPower": data["AerationBasin_TotalAerationPower"].values,
            "VirtualDemand_Electricity_InFlow":  data["VirtualDemand_Electricity_InFlow"].values,
            "VirtualDemand_RestOfFacilityPower": data["VirtualDemand_RestOfFacilityPower"].values,
        }
        # Add battery power if available
        if "BatteryToFacility" in data.columns:
            export_profiles["Battery Power (kW)"] = data["BatteryToFacility"].values

    elif ingest_gas == 'o2':
        # print(f"  HFI500GP (tons/day O2): {data["HFI500GP"].mean():.2f}")
        # print(f"  O2Plant_AerationBasin_Air_Flow: {data['O2Plant_AerationBasin_Air_Flow'].mean():.2f}")
        # print(f"  O2Plant_AerationBasin_Oxygen_Flow: {data['O2Plant_AerationBasin_Oxygen_Flow'].mean():.2f}")
        
        export_profiles = {
            "15 Minute Key": data.index.values,
            "LiftPumpToPrimary": data["LiftPump_PrimaryClarifier_UntreatedSewage_Flow"].values,
            "Blower_AerationBasin_Air_Flow": data["O2Plant_AerationBasin_Oxygen_Flow"].values * 1000 * 1000 / M_O2 / frac_o2_air * R * T_room / (P_ATM * Pa_per_MPa), # TODO: handle in flows_prep
            "AerationBasin_TotalAerationPower": data["O2PlantCompressor_Electricity_OutFlow"].values,
            "VirtualDemand_Electricity_InFlow": data["VirtualDemand_Electricity_InFlow"].values,
            "VirtualDemand_RestOfFacilityPower": data["VirtualDemand_RestOfFacilityPower"].values,
        }

        print(f"  O2Plant_AerationBasin_Air_Flow: {export_profiles['Blower_AerationBasin_Air_Flow'].mean():.2f}")
    
    return export_profiles


def plot_data_stages(processed_data, cleaned_data, imputed_data, final_data, 
                     scaled_data, run_name, ingest_gas, scale_factor):
    """Generate plots for each data processing stage and save them separately"""
    
    # Months to plot - use specific month-year combinations
    if ingest_gas == 'air':
        months_to_plot = [
            (1, 2023),
            (4, 2023),
            (7, 2022),
            (10, 2022)
        ]
    else:
        months_to_plot = [
            (1, 2024),
            (4, 2024),
            (7, 2024),
            (10, 2024)
        ]        
    month_names = {1: "January", 4: "April", 7: "July", 10: "October"}
    
    # print(final_data.columns)
    # Define columns to plot based on gas type
    if ingest_gas == 'o2':
        plot_columns = ["O2PlantCompressor_Electricity_OutFlow", 
                        "VirtualDemand_Electricity_InFlow", 
                        "VirtualDemand_RestOfFacilityPower", 
                        "Cogenerator_VirtualDemand_Electricity_Flow",
                        "PowerGrid1_EBMUD_Electricity_Flow",
                        "PowerGrid2_EBMUD_Electricity_Flow"]
    else:  # air
        plot_columns = ["AerationBasin_TotalAerationPower", 
                        "VirtualDemand_Electricity_InFlow", 
                        "VirtualDemand_RestOfFacilityPower", 
                        "Cogenerator_VirtualDemand_Electricity_Flow",
                        "PowerGrid_SVCW_VirtualDemand_Electricity_Flow",
                        "Battery_Electricity_NetFlow"]
    
    colors = [cb_palette[2],cb_palette[3],cb_palette[0],cb_palette[6],cb_palette[8], cb_palette[4]]
    
    # Create plots for each stage
    stages = [
        ("processed", processed_data),
        ("cleaned", cleaned_data), 
        ("imputed", imputed_data),
        ("final", final_data),
        ("scaled", scaled_data)
    ]
    
    for stage_name, data in stages:
        # For o2 gas, create a summed PowerGrid column
        if ingest_gas == 'o2':
            data = data.copy()
            data['PowerGrid_EBMUD_Electricity_Flow_Sum'] = -(
                data['PowerGrid1_EBMUD_Electricity_Flow'] + 
                data['PowerGrid2_EBMUD_Electricity_Flow']
            )
            # Update plot_columns to use the summed column
            plot_columns_o2 = ["O2PlantCompressor_Electricity_OutFlow", 
                              "VirtualDemand_Electricity_InFlow", 
                              "VirtualDemand_RestOfFacilityPower", 
                              "Cogenerator_VirtualDemand_Electricity_Flow",
                              "PowerGrid_EBMUD_Electricity_Flow_Sum"]
            current_plot_columns = plot_columns_o2
        else:
            current_plot_columns = plot_columns
            
        # Create figure with square subplots to accommodate y-axis labels
        fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        
        for i, (month, year) in enumerate(months_to_plot):
            # Filter data for this specific month-year combination
            month_data = data[(data['month'] == month) & (data['year'] == year)].copy()
            if len(month_data) == 0:
                continue
                
            # hour column for x
            month_data['hour'] = month_data['hour'] + month_data.index.minute / 60
            
            # Calculate mean profiles for all columns to determine y-axis limits
            mean_profiles = {}
            for col in current_plot_columns:
                mean_profiles[col] = month_data.groupby('hour')[col].mean()
            
            # Plot individual columns with custom labels
            # Plot O2PlantCompressor_Electricity_OutFlow
            for date in month_data['date'].unique():
                day_data = month_data[month_data['date'] == date]
                axs[i].plot(day_data['hour'], day_data[current_plot_columns[0]], color=colors[0], 
                            alpha=0.2, linewidth=0.5, zorder=1)
            mean_profile = mean_profiles[current_plot_columns[0]]
            axs[i].plot(mean_profile.index, mean_profile.values, color=colors[0], 
                        linewidth=3.5, label="Aeration Power Load Profile", zorder=3)
            
            # Plot VirtualDemand_Electricity_InFlow
            for date in month_data['date'].unique():
                day_data = month_data[month_data['date'] == date]
                axs[i].plot(day_data['hour'], day_data[current_plot_columns[1]], color=colors[1], 
                            alpha=0.2, linewidth=0.5, zorder=1)
            mean_profile = mean_profiles[current_plot_columns[1]]
            axs[i].plot(mean_profile.index, mean_profile.values, color=colors[1], 
                        linewidth=2.5, label="Virtual Demand", zorder=3)
            
            # Plot VirtualDemand_RestOfFacilityPower
            for date in month_data['date'].unique():
                day_data = month_data[month_data['date'] == date]
                axs[i].plot(day_data['hour'], day_data[plot_columns[2]], color=colors[5], 
                            alpha=0.2, linewidth=0.5, zorder=1)
            mean_profile = mean_profiles[plot_columns[2]]
            axs[i].plot(mean_profile.index, mean_profile.values, color=colors[5], 
                        linewidth=2.5, label="Mean VirtualDemand_RestOfFacilityPower", zorder=3)
            

            # Plot Cogenerator_VirtualDemand_Electricity_Flow
            for date in month_data['date'].unique():
                day_data = month_data[month_data['date'] == date]
                axs[i].plot(day_data['hour'], day_data[current_plot_columns[3]], color=colors[3], 
                            alpha=0.2, linewidth=0.5, zorder=1)
            mean_profile = mean_profiles[current_plot_columns[3]]
            axs[i].plot(mean_profile.index, mean_profile.values, color=colors[3], 
                        linewidth=2.5, label="Cogenerator Power Generation", zorder=3)

            # Plot Grid purchases
            for date in month_data['date'].unique():
                day_data = month_data[month_data['date'] == date]
                axs[i].plot(day_data['hour'], day_data[current_plot_columns[4]], color=colors[2], 
                            alpha=0.2, linewidth=0.5, zorder=1)
            mean_profile = mean_profiles[current_plot_columns[4]]
            axs[i].plot(mean_profile.index, mean_profile.values, color=colors[2], 
                        linewidth=2.5, label="Grid Electricity Purchases", zorder=3)
                        
            # Plot battery power if available
            if len(current_plot_columns) > 5 and current_plot_columns[5] in month_data.columns:
                for date in month_data['date'].unique():
                    day_data = month_data[month_data['date'] == date]
                    axs[i].plot(day_data['hour'], day_data[current_plot_columns[5]], color=colors[4], 
                                alpha=0.2, linewidth=0.5, zorder=1)
                mean_profile = mean_profiles[current_plot_columns[5]]
                axs[i].plot(mean_profile.index, mean_profile.values, color=colors[4], 
                            linewidth=2.5, label="Battery Power", zorder=3)
            
            # y-axis limits
            all_mean_values = []
            for mean_profile in mean_profiles.values():
                all_mean_values.extend(mean_profile.values)
            
            if all_mean_values:
                y_min = min(all_mean_values)
                y_max = max(all_mean_values)
                if ingest_gas == 'air' and scale_factor == 1.0:
                    axs[i].set_ylim(0000, 2000)  # for 13.5 MGD
                elif ingest_gas == 'ο2' and scale_factor == 1.0:
                    axs[i].set_ylim(-4000, 8000)  # for 80 MGD
                else:
                    axs[i].set_ylim(-1000, 5000)  # for 40 MGD
            
            # Set subplot properties
            axs[i].set_title(f"{month_names[month]} {year}", fontsize=20)
            axs[i].set_xlabel("Hour of Day", fontsize=18)
            axs[i].set_ylabel("Power (kW)", fontsize=18)  # Add y-label to all subplots
            axs[i].axhline(0, color='k', alpha=0.3)
            axs[i].set_xlim(0, 24)
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            
            # Show y-axis tick labels on all subplots
            axs[i].tick_params(axis='y', which='major', labelsize=12)
            # Ensure y-axis labels are visible
            axs[i].yaxis.set_tick_params(labelleft=True)
            
            # Make subplot area more square to accommodate y-axis labels
            axs[i].set_aspect('auto', adjustable='box')
    
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.8, 1.6), 
                  fontsize=14, ncol=2, frameon=False)
    
    # Adjust layout to prevent cropping of x-axis labels and legend
    plt.subplots_adjust(top=0.7, bottom=0.2, wspace=0.4)  # Increase top margin for legend and add space between subplots
    outdir = f"aeration_flexibility/output_plots/{run_name}/paper_figures"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{run_name}_{stage_name}_daily_profiles.png", dpi=200)
    plt.close()


# Configuration for different gas types
gas_configs = {
    'air': {
        'network_file': "network.json",
        'params_file': "parameters.json", 
        'facility_id': "SVCW",
        'source_files': ["20220701_20240318_SVCW_RawData.csv", "battery_power_data.csv"],
        'date_range': ["2022-07-01", "2024-03-18"]
    },
    'o2': {
        'network_file': "ebmud_network.json",
        'params_file': "ebmud_parameters.json",
        'facility_id': "EBMUD", 
        'source_files': ["scada_combined_data.csv"],
        'date_range': ["2024-01-01", "2024-12-31"]
    }
}
def ingest_data(run_name="run_name", ingest_gas='air', scale_factor=None, shorten_months_run=1):
    """Ingest and process data for oxygen storage analysis"""
    
    # Create output directory and clean data
    os.makedirs(f"aeration_flexibility/output_data/{run_name}", exist_ok=True)
    config = gas_configs[ingest_gas]
    scaled_data, final_data = clean_data_with_flows_prep(
        ingest_gas, config['network_file'], config['params_file'], config['facility_id'],
        config['source_files'], config['date_range'], run_name=run_name, 
        scale_factor=scale_factor
    )
    
    # Setup datetime index with timezone handling
    scaled_data = scaled_data.reset_index()
    if "DateTime" in scaled_data.columns:
        scaled_data["DateTime"] = pd.to_datetime(scaled_data["DateTime"])
        scaled_data = scaled_data.dropna(subset=["DateTime"])
        scaled_data = scaled_data.set_index("DateTime")
    
    # Ensure proper datetime index with timezone
    if not isinstance(scaled_data.index, pd.DatetimeIndex):
        scaled_data.index = pd.to_datetime(scaled_data.index)
    
    # Handle timezone conversion
    if scaled_data.index.tz is None:
        scaled_data.index = scaled_data.index.tz_localize('US/Pacific', ambiguous='NaT', nonexistent='shift_forward')
    else:
        scaled_data.index = scaled_data.index.tz_convert('US/Pacific')
    
    # Drop ambiguous times and add time-based columns
    scaled_data = scaled_data[~scaled_data.index.isna()]
    scaled_data = scaled_data.assign(
        hour=scaled_data.index.hour,
        month=scaled_data.index.month,
        year=scaled_data.index.year,
        date=scaled_data.index.date,
        **{'15_minute_key': scaled_data.index.hour * 4 + scaled_data.index.minute // 15}
    )
    
    # Create day profiles for complete days
    valid_dates = sorted(set(d for d in scaled_data.index.date if pd.notnull(d)))
    print(scaled_data.columns)
    day_profiles = {
        date.strftime("%Y-%m-%d"): {
            "profile": get_export_profiles_standardized(
                scaled_data.loc[scaled_data.index.date == date], ingest_gas
            ),
            "month": date.month,
            "year": date.year,
        }
        for date in valid_dates
        if len(scaled_data.loc[scaled_data.index.date == date]) == 96
    }
    
    # Shorten months if requested
    if shorten_months_run:
        # Group by month-year and select first n months
        month_year_groups = {}
        for date_str, day_data in day_profiles.items():
            key = f"{day_data['year']}-{day_data['month']:02d}"
            month_year_groups.setdefault(key, []).append(date_str)
        
        selected_months = sorted(month_year_groups.keys())[:shorten_months_run]
        day_profiles = {
            date_str: day_profiles[date_str]
            for month_key in selected_months
            for date_str in month_year_groups[month_key]
        }
    
    # Save day profiles
    with open(f"aeration_flexibility/output_data/{run_name}/day_profiles.pkl", "wb") as f:
        pickle.dump(day_profiles, f)