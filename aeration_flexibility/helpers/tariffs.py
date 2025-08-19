import pandas as pd
import numpy as np
import os
from electric_emission_cost import costs
from electric_emission_cost.costs import parametrize_rate_data
import random

# Set random seed for reproducibility
RANDOM_SEED = 4
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define calendar months and their date ranges
CALENDAR_MONTHS = {
    "2022-01": {"start": "2022-01-01", "end": "2022-02-01"},
    "2022-02": {"start": "2022-02-01", "end": "2022-03-01"},
    "2022-03": {"start": "2022-03-01", "end": "2022-04-01"},
    "2022-04": {"start": "2022-04-01", "end": "2022-05-01"},
    "2022-05": {"start": "2022-05-01", "end": "2022-06-01"},
    "2022-06": {"start": "2022-06-01", "end": "2022-07-01"},
    "2022-07": {"start": "2022-07-01", "end": "2022-08-01"},
    "2022-08": {"start": "2022-08-01", "end": "2022-09-01"},
    "2022-09": {"start": "2022-09-01", "end": "2022-10-01"},
    "2022-10": {"start": "2022-10-01", "end": "2022-11-01"},
    "2022-11": {"start": "2022-11-01", "end": "2022-12-01"},
    "2022-12": {"start": "2022-12-01", "end": "2023-01-01"},
    "2023-01": {"start": "2023-01-01", "end": "2023-02-01"},
    "2023-02": {"start": "2023-02-01", "end": "2023-03-01"},
    "2023-03": {"start": "2023-03-01", "end": "2023-04-01"},
    "2023-04": {"start": "2023-04-01", "end": "2023-05-01"},
    "2023-05": {"start": "2023-05-01", "end": "2023-06-01"},
    "2023-06": {"start": "2023-06-01", "end": "2023-07-01"},
    "2023-07": {"start": "2023-07-01", "end": "2023-08-01"},
    "2023-08": {"start": "2023-08-01", "end": "2023-09-01"},
    "2023-09": {"start": "2023-09-01", "end": "2023-10-01"},
    "2023-10": {"start": "2023-10-01", "end": "2023-11-01"},
    "2023-11": {"start": "2023-11-01", "end": "2023-12-01"},
    "2023-12": {"start": "2023-12-01", "end": "2024-01-01"},
    "2024-01": {"start": "2024-01-01", "end": "2024-02-01"},
    "2024-02": {"start": "2024-02-01", "end": "2024-03-01"},
    "2024-03": {"start": "2024-03-01", "end": "2024-04-01"},
    "2024-04": {"start": "2024-04-01", "end": "2024-05-01"},
    "2024-05": {"start": "2024-05-01", "end": "2024-06-01"},
    "2024-06": {"start": "2024-06-01", "end": "2024-07-01"},
    "2024-07": {"start": "2024-07-01", "end": "2024-08-01"},
    "2024-08": {"start": "2024-08-01", "end": "2024-09-01"},
    "2024-09": {"start": "2024-09-01", "end": "2024-10-01"},
    "2024-10": {"start": "2024-10-01", "end": "2024-11-01"},
    "2024-11": {"start": "2024-11-01", "end": "2024-12-01"},
    "2024-12": {"start": "2024-12-01", "end": "2025-01-01"},
}

# Generate replacement days for each calendar day
REPLACEMENT_DAYS = {}
for month_key, month_info in CALENDAR_MONTHS.items():
    year, month = map(int, month_key.split("-"))
    
    # Use the same leap year logic as get_all_days_in_month
    if month == 2:
        # Leap year if divisible by 4, but not by 100 unless also divisible by 400
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_month = 29 if is_leap else 28
    elif month in [1, 3, 5, 7, 8, 10, 12]:
        days_in_month = 31
    else:
        days_in_month = 30
    
    all_days = [f"{year}-{month:02d}-{day:02d}" for day in range(1, days_in_month + 1)]

    for day in all_days:
        # Create a list of replacement days, excluding the current day
        other_days = [d for d in all_days if d != day]
        REPLACEMENT_DAYS[day] = random.sample(other_days, len(other_days))


def get_replacement_days(date):
    """Get the predefined list of replacement days for a given day."""
    return REPLACEMENT_DAYS.get(date, [])


def get_month_dates(year, month):
    """Get start and end dates for a specific month."""
    month_key = f"{year}-{month:02d}"
    return {
        "start": np.datetime64(CALENDAR_MONTHS[month_key]["start"]),
        "end": np.datetime64(CALENDAR_MONTHS[month_key]["end"]),
    }


def get_all_days_in_month(year, month):
    """Get all days in a month as a list of date strings."""
    # Handle leap years for February
    if month == 2:
        # Leap year if divisible by 4, but not by 100 unless also divisible by 400
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_month = 29 if is_leap else 28
    elif month in [1, 3, 5, 7, 8, 10, 12]:
        days_in_month = 31
    else:
        days_in_month = 30
    
    return [f"{year}-{month:02d}-{day:02d}" for day in range(1, days_in_month + 1)]


# Get the path to the data directory
base_dir = os.path.dirname(os.path.dirname(__file__))


def generate_parametrized_tariffs():
    """
    Automatically generate parametrized tariff files from the base billing.csv file.
    Creates variants with different demand/energy scaling and peak window shifts.
    """
    # Define the parametrization variants based on the existing files
    variants = [
        {"scale_all_demand": 1.0, "variant_name": "svcw_d1.0_e1.0_w0"},

        # Demand scaling variants (d0.9, d1.0, d1.1)
        {"scale_all_demand": 0.9, "variant_name": "svcw_d0.9_e1.0_w0"},
        {"scale_all_demand": 1.1, "variant_name": "svcw_d1.1_e1.0_w0"},
        
        # Energy scaling variants (e0.9, e1.0, e1.1)
        {"scale_all_energy": 0.9, "variant_name": "svcw_d1.0_e0.9_w0"},
        {"scale_all_energy": 1.1, "variant_name": "svcw_d1.0_e1.1_w0"},
        
        # Peak window shift variants (wneg1, w1)
        {"shift_peak_hours_before": -0.25, "shift_peak_hours_after": 0.25, "variant_name": "svcw_d1.0_e1.0_wneg0.25"},
        {"shift_peak_hours_before": 0.25, "shift_peak_hours_after": -0.25, "variant_name": "svcw_d1.0_e1.0_w0.25"},
        
        # ALL demand/energy scaling variants
        {"scale_all_demand": 1.1, "variant_name": "svcw_alldemand1.1"},
        {"scale_all_demand": 0.9, "variant_name": "svcw_alldemand0.9"},
        {"scale_all_energy": 1.1, "variant_name": "svcw_allenergy1.1"},
        {"scale_all_energy": 0.9, "variant_name": "svcw_allenergy0.9"},
    ]
    
    # Path to the base billing files
    billing_base_files = ["svcw.csv", "ebmud.csv"]
    for file in billing_base_files:
        utility = file.replace(".csv", "")  # Extract utility name from filename
        base_billing_path = os.path.join(base_dir, f"data/billing/{file}")
        billing_dir = os.path.join(base_dir, "data/billing")
        
        # Ensure the billing directory exists
        os.makedirs(billing_dir, exist_ok=True)
        
        # Load the base billing data
        if not os.path.exists(base_billing_path):
            print(f"Warning: Base billing file not found at {base_billing_path}")
            continue
        
        base_data = pd.read_csv(base_billing_path)
        print(f"Loaded base billing data from {base_billing_path}")
        
        # Generate each variant for this utility
        for variant in variants:
            variant_name = variant["variant_name"]
            # Replace the utility prefix in the variant name
            if variant_name.startswith("svcw_"):
                variant_name = variant_name.replace("svcw_", f"{utility}_")
            elif variant_name.startswith("ebmud_"):
                variant_name = variant_name.replace("ebmud_", f"{utility}_")
            else:
                # If no prefix, add the utility prefix
                variant_name = f"{utility}_{variant_name}"
            
            output_path = os.path.join(billing_dir, f"{variant_name}.csv")
            
            parametrized_data = parametrize_rate_data(base_data.copy(), **variant)
            parametrized_data.to_csv(output_path, index=False)
            print(f"Generated {variant_name}.csv")


def ensure_parametrized_tariffs_exist():
    """
    Check if parametrized tariff files exist, and generate them if they don't.
    This ensures the required tariff variants are available before loading.
    """
    billing_dir = os.path.join(base_dir, "data/billing")
    required_files = []
    for utility in ["svcw", "ebmud"]:
        required_files.extend([
            f"{utility}_d1.0_e1.0_w0.csv",  # Base file
            f"{utility}_d0.9_e1.0_w0.csv",
            f"{utility}_d1.1_e1.0_w0.csv", 
            f"{utility}_d1.0_e0.9_w0.csv",
            f"{utility}_d1.0_e1.1_w0.csv",
            f"{utility}_d1.0_e1.0_wneg0.25.csv",
            f"{utility}_d1.0_e1.0_w0.25.csv",
            f"{utility}_alldemand1.1.csv",
            f"{utility}_alldemand0.9.csv",
            f"{utility}_allenergy1.1.csv",
            f"{utility}_allenergy0.9.csv"
        ])
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(billing_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        print(f"Missing parametrized tariff files: {missing_files}")
        print("Generating parametrized tariffs...")
        generate_parametrized_tariffs()


def load_tariff_data():
    """
    Load all tariff CSV files from the billing directory.
    Returns a dictionary mapping tariff keys to DataFrames.
    """
    # Ensure parametrized tariff files exist before loading
    ensure_parametrized_tariffs_exist()
    
    billing_dir = os.path.join(base_dir, "data/billing")
    tariff_data = {}
    
    for fname in os.listdir(billing_dir):
        if fname.endswith(".csv") and not fname.startswith("."):
            base = fname[:-4]  # Remove .csv extension
            fpath = os.path.join(billing_dir, fname)
            try:
                if base == "billing":
                    # Add both 0.0__billing (base) and 1.0__billing (solar)
                    tariff_data["0.0__billing"] = pd.read_csv(fpath)
                    tariff_data["1.0__billing"] = pd.read_csv(fpath)
                else:
                    tariff_data[f"0.0__{base}"] = pd.read_csv(fpath)
                    tariff_data[f"1.0__{base}"] = pd.read_csv(fpath)
            except Exception as e:
                print(f"Warning: Could not load {fname}: {e}")
    
    return tariff_data


def build_tariff_configs(reference_file, tariff_data):
    """Build tariff_configs using the reference file for county/state descriptions."""
    ref_df = pd.read_csv(reference_file, dtype=str)
    # Convert CURRENT_DESIGN_FLOW to float for sorting
    ref_df["CURRENT_DESIGN_FLOW"] = pd.to_numeric(
        ref_df["CURRENT_DESIGN_FLOW"], errors="coerce"
    )

    # Build a mapping from CWNS_ID to (County, State, Design Flow)
    cwns_to_info = {}
    for _, row in ref_df.iterrows():
        cwns = row["CWNS_ID"]
        county = row["COUNTY_NAME"]
        state = row["STATE_CODE"]
        design_flow = row["CURRENT_DESIGN_FLOW"]
        if pd.notnull(county) and pd.notnull(state):
            cwns_to_info[cwns] = {
                "location": f"{county} County, {state}",
                "design_flow": (float(design_flow) if pd.notnull(design_flow) else 0),
            }

    # Build the configs with design flow information
    configs = {}
    configs_with_flow = []

    for key in tariff_data.keys():
        # Remove multiplier prefix for lookup
        base_key = key.split("__", 1)[-1].replace("___", "_")
        if base_key == "billing":
            if key.startswith("1.0__"):
                configs[key.replace("___", "_")] = (
                    "Redwood City, CA\nSolar for Storage)"
                )
                configs_with_flow.append(
                    (
                        key.replace("___", "_"),
                        "Redwood City, CA\nSolar for Storage)",
                        float("inf"),
                    )
                )
            else:
                configs[key.replace("___", "_")] = "Redwood City, CA"
                configs_with_flow.append(
                    (key.replace("___", "_"), "Redwood City, CA", float("inf"))
                )
        else:
            if key.startswith("1.0__"):
                cwns_id = base_key.split("_")[0]
                info = cwns_to_info.get(
                    cwns_id,
                    {
                        "location": "Unknown County, Unknown State",
                        "design_flow": 0,
                    },
                )
                configs[key.replace("___", "_")] = (
                    info["location"] + "\nSolar for Storage"
                )
                configs_with_flow.append(
                    (
                        key.replace("___", "_"),
                        info["location"],
                        info["design_flow"],
                    )
                )
            else:
                cwns_id = base_key.split("_")[0]
                info = cwns_to_info.get(
                    cwns_id,
                    {
                        "location": "Unknown County, Unknown State",
                        "design_flow": 0,
                    },
                )
                configs[key.replace("___", "_")] = info["location"]
                configs_with_flow.append(
                    (
                        key.replace("___", "_"),
                        info["location"],
                        info["design_flow"],
                    )
                )
    
    # Sort configs by solar multiplier first (0.0 before 1.0), then by design flow in descending order
    configs_with_flow.sort(key=lambda x: (x[0].startswith("1.0"), -x[2]))

    return configs


# Global variables for lazy loading
_tariff_data = None
_tariff_configs = None
_tariff_config_keys = None


def get_tariff_data():
    """Get tariff data (lazy loading)."""
    global _tariff_data
    if _tariff_data is None:
        _tariff_data = load_tariff_data()
    return _tariff_data


def get_tariff_configs():
    """Get tariff configs (lazy loading)."""
    global _tariff_configs
    if _tariff_configs is None:
        tariff_data = get_tariff_data()
        _tariff_configs = build_tariff_configs(
        os.path.join(base_dir, "data/billing/reference/pure_oxygen_facilities.csv"),
            tariff_data,
        )
    return _tariff_configs


def get_tariff_config_keys():
    """Get tariff config keys (lazy loading)."""
    global _tariff_config_keys
    if _tariff_config_keys is None:
        configs = get_tariff_configs()
        _tariff_config_keys = list(configs.keys())
    return _tariff_config_keys


def get_charge_dict_for_day(tariff_data, date):
    """
    Get charge dict for a specific day.
    Args:
        tariff_data: DataFrame containing the tariff rate data
        date: Date string in format 'YYYY-MM-DD'
    Returns:
        Dictionary of charge arrays for the day
    """
    # Convert date to datetime64
    start_date = np.datetime64(date)
    end_date = start_date + np.timedelta64(1, 'D')  # Add 1 day
    
    return costs.get_charge_dict(
        start_date, end_date, tariff_data, resolution="15m"
    )


def get_charge_dict_for_month(tariff_data, year, month):
    """
    Get charge dict for a specific month.
    Args:
        tariff_data: DataFrame containing the tariff rate data
        year: Year (int)
        month: Month (int)
    Returns:
        Dictionary of charge arrays for the month
    """
    month_dates = get_month_dates(year, month)
    return costs.get_charge_dict(
        month_dates["start"], month_dates["end"], tariff_data, resolution="15m"
    )