import numpy as np
import pickle
import pandas as pd
import os
import gc
import time
import copy
from multiprocessing import Pool
from helpers.operational_optimization_problem import *
from helpers.tariffs import *
from helpers.capex_npv_energy_metrics import *
from helpers.parameters import *
from helpers.config_labels import *
from helpers.tariffs import get_tariff_data, get_charge_dict_for_month
from electric_emission_cost import costs, metrics

from helpers.tariffs import CALENDAR_MONTHS

# Set numpy random seed to match tariffs.py
np.random.seed(RANDOM_SEED)

def concatenate_n_days_data(baseline_val_dict, charge_dict_month, days, start_day_idx):
    """
    Concatenate data for N consecutive days starting from start_day_idx.
    
    Args:
        baseline_val_dict: Dictionary of baseline values for each day
        charge_dict_month: Monthly charge dictionary
        days: List of days in the month
        start_day_idx: Starting index for the N-day group
    
    Returns:
        tuple: (concatenated_baseline_vals, concatenated_charge_dict, day_dates)
    """
    # Get the N days (or remaining days if < N)
    end_day_idx = min(start_day_idx + len(days), len(days))
    group_days = days[start_day_idx:end_day_idx]
    
    # Concatenate baseline values
    concatenated_baseline_vals = {}
    for key in baseline_val_dict[group_days[0]].keys():
        if isinstance(baseline_val_dict[group_days[0]][key], np.ndarray):
            # For indexed parameters, concatenate arrays and convert to pandas Series
            concatenated_vals = []
            for day in group_days:
                concatenated_vals.append(baseline_val_dict[day][key])
            concatenated_baseline_vals[key] = pd.Series(np.concatenate(concatenated_vals))
        else:
            # For non-indexed parameters, use the first day's value
            concatenated_baseline_vals[key] = baseline_val_dict[group_days[0]][key]
    
    # Concatenate charge dictionary
    concatenated_charge_dict = {}
    for charge_key in charge_dict_month.keys():
        concatenated_vals = []
        for day in group_days:
            _, _, day_num = map(int, day.split("-"))
            start_idx = (day_num - 1) * 96
            end_idx = start_idx + 96
            concatenated_vals.append(charge_dict_month[charge_key][start_idx:end_idx])
        concatenated_charge_dict[charge_key] = np.concatenate(concatenated_vals)
    
    return concatenated_baseline_vals, concatenated_charge_dict, group_days


def extract_day_results_from_n_day_profile(profile, day_idx, total_days_in_group):
    """
    Extract results for a specific day from an N-day optimization profile.
    
    Args:
        profile: N-day optimization profile (numpy arrays for time-series data)
        day_idx: Index of the day to extract (0, 1, ..., N-1)
        total_days_in_group: Total number of days in the group
    
    Returns:
        dict: Single-day profile with pandas Series for time-series data
    """
    if not profile:
        return None    
    
    day_profile = {}
    start_idx = day_idx * 96
    for key, value in profile.items():
        # Check if this is time-series data
        if hasattr(value, '__len__') and len(value) == total_days_in_group * 96:
            day_profile[key] = pd.Series(value[start_idx:start_idx + 96])
        else:  # non-time-series data
            day_profile[key] = value
    
    return day_profile

def process_n_days(days, design_key, base_wwtp_key, upgrade_key, tariff_key, new_param_vals, 
                  baseline_val_dict, charge_dict_month, prev_demand_dict=None,
                  initial_storage_state=None, horizon_days=3):
    """
    Process N consecutive days as a single optimization problem.
    
    Args:
        days: List of N consecutive days to process
        design_key: Design key for the optimization
        base_wwtp_key: Base WWTP configuration key
        upgrade_key: Upgrade configuration key
        tariff_key: Tariff key
        new_param_vals: System new_param_vals
        baseline_val_dict: Dictionary of baseline values for each day
        charge_dict_month: Monthly charge dictionary
        prev_demand_dict: Previous demand dictionary
        initial_storage_state: Initial storage state for the first day
    
    Returns:
        tuple: (profiles_dict, new_param_vals, max_values_dict, initial_storage_states_dict)
    """
    
    # Load tariff data for this tariff_key
    tariff_data = get_tariff_data()
    if tariff_key not in tariff_data:
        print(f"Warning: Tariff key {tariff_key} not found in tariff data")
        return (None, None, None, None)
    
    # Concatenate data for the N days
    concatenated_baseline_vals, concatenated_charge_dict, group_days = concatenate_n_days_data(
        baseline_val_dict, charge_dict_month, days, 0
    )

    # Create problem instance for N days
    problem_start = time.time()
    problem = O2Problem(
        single_day_config=f"{base_wwtp_key}___{upgrade_key}___{tariff_key}",
        design_key=design_key, date=f"{group_days[0]}_to_{group_days[-1]}", 
        initial_storage_state=initial_storage_state, data_wwtp='svcw', horizon_days=horizon_days
    )

    # Check if concatenated data has correct length
    if len(concatenated_baseline_vals["Ndot_target"]) != len(group_days) * 96:
        print(f"Skipping {group_days[0]} to {group_days[-1]} - Ndot_target has {len(concatenated_baseline_vals['Ndot_target'])} entries")
        del problem
        return (None, None, None, None)

    # Construct and solve optimization problem with N-day data
    construct_start = time.time()
    problem.construct_problem(
        charge_dict=concatenated_charge_dict,
        baseline_vals=concatenated_baseline_vals,
        new_param_vals=new_param_vals,
        prev_demand_dict=prev_demand_dict,
    )
    # print(f"  Problem construction: {(time.time() - construct_start):.3f}s")
    
    solve_start = time.time()
    profile = problem.solve_optimization_day()
    # print(f"  Problem Solve time: {(time.time() - solve_start):.3f}s, Total: {(time.time() - problem_start):.3f}s")
    
    # problem.print_cost_values(charge_dict=concatenated_charge_dict, prev_demand_dict=prev_demand_dict)
    del problem.m
    del problem
    
    if profile and "Edot_t_net" in profile:
        profiles_dict = {}
        max_values_dict = {}
        
        for day_idx, day in enumerate(group_days):
            # Extract day-specific profile
            day_profile = extract_day_results_from_n_day_profile(
                profile, day_idx, len(group_days)
            )
            
            if day_profile and "Edot_t_net" in day_profile:
                profiles_dict[day] = day_profile
                
                # Calculate max values for this day
                max_values_dict[day] = {
                    "Ndot_c": np.max(day_profile["Ndot_c"]) if "Ndot_c" in day_profile else 0,
                    "Edot_c": np.max(day_profile["Edot_c"]) if "Edot_c" in day_profile else 0,
                }
                
                # Store initial storage state for the second to last day of the group
                if day_idx == len(group_days) - 2 or len(group_days)<horizon_days:  # Second to last day
                    initial_storage_state_next_run = extract_final_storage_state(day_profile)
            else:
                profiles_dict[day] = None
                max_values_dict[day] = {"Ndot_c": 0, "Edot_c": 0}
                initial_storage_state_next_run = None
        
        return profiles_dict, max_values_dict, initial_storage_state_next_run

    return None, None, None


def extract_final_storage_state(profile):
    """
    Extract the final storage state from a profile for use as initial state in the next day.
    
    Args:
        profile: Dictionary containing the optimization profile with storage variables
        
    Returns:
        dict: Dictionary containing the final storage state with keys:
            - 'N': Final moles of O2 in storage (for tank systems)
            - 'E': Final energy in storage (for battery systems)
    """
    if not profile:
        return None
    
    final_storage_state = {}
    
    # Extract initial storage moles (N) for tank systems
    if 'N' in profile and len(profile['N']) > 0:
        final_storage_state['N'] = profile['N'].iloc[-1]  # Last value in the series
    
    # Extract initial energy (E) for battery systems
    if 'E' in profile and len(profile['E']) > 0:
        final_storage_state['E'] = profile['E'].iloc[-1]  # First value in the series
    
    return final_storage_state if final_storage_state else None


def check_design_feasibility(month_results, summer_multiplier):
    """
    Centralized function to check if a design is feasible based on month results.
    
    Args:
        month_results: Dictionary of {day: (profile, new_param_vals, max_values)}
        summer_multiplier: Summer multiplier value
    
    Returns:
        tuple: (is_feasible, failure_reason, failed_days_count)
    """
    if not month_results:
        return False, "no_results", 0
    
    failed_days = sum(1 for profile, _ in month_results.values() 
                     if not profile or "Edot_t_net" not in profile)
    
    # Check for insufficient storage
    if float(summer_multiplier) > 1.0 and failed_days > 5: # more than 2 horizons
        return False, "insufficient_storage", failed_days
    
    # Check for general failure (more than 50% failed)
    if failed_days > len(month_results) / 2:
        return False, "half_failed", failed_days
    
    return True, None, failed_days

def skip_saving_last_day(horizon_days, i, day_idx, days):
    is_last_day_of_month = (i + horizon_days >= len(days))
    is_last_day_of_horizon = (day_idx == horizon_days - 1)
    return is_last_day_of_horizon and not is_last_day_of_month

def process_design_point_and_month(design_point_data, run_name, intermediate_filepaths=None, horizon_days=3):
    """Process a single design point and month combination using N-day optimization with 1-day overlap.

    Args:
        design_point_data: Tuple of (design_key, month_key, month_days, base_wwtp_key, upgrade_key,
                                    tariff_key, new_param_vals, baseline_val_dict, new_param_vals)
        run_name: Name of the run for saving intermediate results
        intermediate_filepaths: Dictionary of file paths to already-solved intermediate results
        horizon_days: Number of days to process in each horizon (default 3)
    """
    (design_key, month_key, month_days, base_wwtp_key, upgrade_key,
     tariff_key, new_param_vals, baseline_val_dict, new_param_vals) = design_point_data    

    # Process the month using N-day optimization with 1-day overlap
    month_start_time = time.time()
    year, month = month_key.split("-")
    
    print(f"processing {design_key} in month {month}, {year}")

    # Initialize prev_demand_dict with zero demands for all charge types
    prev_demand_dict = {}
    demand_charge_info_dict = get_demand_charge_info_dict(
        tariff_key, month, int(year)
    )
    for charge_key in demand_charge_info_dict.keys():
        if "demand" in charge_key:
            prev_demand_dict[charge_key] = {"demand": 0.0, "cost": 0.0}

    # Create directory for intermediate results
    _, _, summer_multiplier, summer_smoothing = split_key(base_wwtp_key)
    if isinstance(summer_smoothing, str) and summer_smoothing.startswith("neg"):
        summer_smoothing_int = -int(summer_smoothing[3:])  # Remove "neg" prefix and convert to negative
    else:
        summer_smoothing_int = int(summer_smoothing)
    suffix = get_summer_key(summer_multiplier, summer_smoothing_int)
    config_name = get_config_name(base_wwtp_key, upgrade_key, tariff_key, suffix)
    intermediate_dir = f"aeration_flexibility/output_data/{run_name}/intermediate/{config_name}"
    os.makedirs(intermediate_dir, exist_ok=True)    
    
    # Load tariff data for the month
    tariff_data = get_tariff_data()
    if tariff_key not in tariff_data:
        print(f"Warning: Tariff key {tariff_key} not found in tariff data")
        return (design_key, month_key, [], new_param_vals, {"baseline": [], "optimized": []}, 
                False, "tariff_not_found")
    
    charge_dict_month = get_charge_dict_for_month(tariff_data[tariff_key], int(year), int(month))

    # Process days in overlapping groups with 1-day overlap
    processed_days = set()
    month_results = {}  # Store the final result for each day
    i = 0
    day_failures = 0
    while i < len(month_days):
        group_days = month_days[i:i+horizon_days]
        
        # Check if we have intermediate results for all days in this group
        all_intermediate_available = True
        if intermediate_filepaths and design_key in intermediate_filepaths and month_key in intermediate_filepaths[design_key]:
            for group_day in group_days:
                if group_day not in intermediate_filepaths[design_key][month_key]:
                    all_intermediate_available = False
                    break
        else:
            all_intermediate_available = False
        
        if all_intermediate_available:
            dummy_file_days = []
            for group_day in group_days:
                with open(intermediate_filepaths[design_key][month_key][group_day], 'rb') as f:
                    group_day_data = pickle.load(f)
                
                profile = group_day_data["profile"]
                max_values = group_day_data["max_values"]
                
                if profile and "Edot_t_net" in profile:  # solved day
                    month_results[group_day] = (profile, max_values)
                    processed_days.add(group_day)
                else:  # dummy file / failed day
                    month_results[group_day] = (None, max_values)
                    day_failures += 1
                    dummy_file_days.append(group_day)
                
                del group_day_data
            if dummy_file_days:
                print(f"  Found dummy files for days: {dummy_file_days}")
        else:
            initial_storage_state = None
            if i == 0:
                if float(summer_multiplier) > 1.0: # Give half storage to help with first day feasibility
                    initial_storage_state = {'N': new_param_vals.get('N_max',0)/2, 'E': 0}
                else:  # O2Problem will use minimum values
                    pass
            elif i > 0:
                # Use initial storage state from the last day of the previous group
                prev_group_last_day = month_days[i-1]  # This was the last day in the previous group
                if prev_group_last_day in month_results:
                    prev_profile = month_results[prev_group_last_day][0]
                    if prev_profile:
                        # Get the initial storage state (first timestep) of the last day from previous group
                        initial_storage_state = extract_final_storage_state(prev_profile)
            
            # Process the N-day group
            profiles_dict, max_values_dict, final_storage_state_prev_run = process_n_days(
                group_days, design_key, base_wwtp_key, upgrade_key, tariff_key, new_param_vals,
                baseline_val_dict, charge_dict_month, prev_demand_dict=prev_demand_dict,
                initial_storage_state=initial_storage_state, horizon_days=horizon_days
            )
            
            if profiles_dict:
                # Save results for each day not already processed
                for solved_day_idx, solved_day in enumerate(group_days):
                    # Skip saving the last day of the horizon unless it's the last day of the month
                    if skip_saving_last_day(horizon_days, i, solved_day_idx, month_days):
                        continue
                    
                    profile = profiles_dict[solved_day]
                    max_values = max_values_dict[solved_day]
                    
                    # Save intermediate results for this day
                    intermediate_data = {
                        "profile": profile,
                        "new_param_vals": new_param_vals,
                        "max_values": max_values,
                        "prev_demand_dict": prev_demand_dict,
                        "initial_storage_state": final_storage_state_prev_run,
                    }
                    intermediate_file = os.path.join(
                        intermediate_dir, f"{design_key}_{month_key}_{solved_day}.pkl"
                    )
                    intermediate_file = os.path.abspath(intermediate_file)
                    with open(intermediate_file, "wb") as f:
                        pickle.dump(intermediate_data, f)
                    
                    # Store final result for this day
                    month_results[solved_day] = (profile, max_values)
                    processed_days.add(solved_day)
                    
            else:
                # N-day group failed
                print("Saving dummy files to prevent re-solving")
                for failed_day_idx, failed_day in enumerate(group_days):
                    if skip_saving_last_day(horizon_days, i, failed_day_idx, month_days):
                        continue
                    dummy_data = {
                        "profile": None,
                        "new_param_vals": new_param_vals,
                        "max_values": {"Ndot_c": 0, "Edot_c": 0},
                        "prev_demand_dict": prev_demand_dict,
                        "initial_storage_state": initial_storage_state,  # Preserve the storage state from previous horizon
                        "failure_reason": "infeasible"
                    }
                    
                    intermediate_file =  os.path.abspath(os.path.join(
                        intermediate_dir, f"{design_key}_{month_key}_{failed_day}.pkl"
                    ))
                    with open(intermediate_file, "wb") as f:
                        pickle.dump(dummy_data, f)
                    
                    month_results[failed_day] = (None, {"Ndot_c": 0, "Edot_c": 0})
                    day_failures += 1
        
        # Update prev_demand_dict with maximum demands from processed days
        for processed_day in processed_days:
            profile = month_results[processed_day][0]
            day_max_dict = get_max_dict_from_profile(profile, demand_charge_info_dict)
            if day_max_dict is None:
                continue  # no updates to prev_demand_dict
            else:
                for charge_key, charge_values in day_max_dict.items():
                    if charge_values["demand"] > prev_demand_dict[charge_key]["demand"]:
                        prev_demand_dict[charge_key] = charge_values.copy()
        
        # Check for early termination due to infeasibility
        if day_failures > 0:
            is_feasible, failure_reason, _ = check_design_feasibility(month_results, summer_multiplier)
            if not is_feasible:
                print(f"  ✗ Design {design_key} in {month_key} marked as {failure_reason} after {day_failures} failures")
                return (design_key, month_key, month_results, new_param_vals, False, failure_reason)
        
        i += (horizon_days - 1)  # include 1 day overlap for next group

    is_feasible, failure_reason, _ = check_design_feasibility(month_results, summer_multiplier)

    print(f"Month {month_key} for design key {design_key} completed in {time.time() - month_start_time:.3f}s")

    return (
        design_key,
        month_key,
        month_results,
        new_param_vals,
        is_feasible,
        failure_reason,
    )

def is_valid_npv(npv):
    """Check if NPV is a valid finite number."""
    if isinstance(npv, (np.ndarray, list)):
        return np.isfinite(npv).all()
    return np.isfinite(npv)


def sample_from_multivariate_normal(all_designs, designs_per_run, o2_range, comp_ratio_range, upgrade_key,
                                    elite_samples=8, exp_weight=2.0, min_variance=1e-3):
    """
    Sample new points using cross-entropy method with configurable hyperparameters.
    
    Args:
        all_designs: Dictionary of existing design results
        designs_per_run: Number of new points to generate
        o2_range: (min, max) range for O2 hours
        comp_ratio_range: (min, max) range for compression ratio
        upgrade_key: Type of upgrade (determines if univariate or bivariate)
        elite_samples: Number of elite points to use for fitting distribution
        prior_improvement: Improvement from previous iteration (None for first iteration)
        exp_weight: Exponential weighting factor for elite selection
        min_variance: Minimum variance for numerical stability
    """
    # Local random state for consistent sampling from seed
    local_rng = np.random.RandomState(RANDOM_SEED)
    
    # Extract valid points and NPVs
    points, npvs = [], []
    is_univariate = "battery" in upgrade_key or "liquid" in upgrade_key
    
    for design_key, design_data in all_designs.items():
        npv = design_data["metrics"]["npv"]["total"]
        if is_valid_npv(npv):
            coords = [float(x) for x in design_key.split("__")]
            points.append(coords[:1] if is_univariate else coords[:2])
            npvs.append(npv)
    
    if not points:
        print('No points found')
        return [], None
    
    points, npvs = np.array(points), np.array(npvs)
    
    # Select elite points
    elite_count = min(elite_samples, len(points))
    elite_indices = np.argsort(npvs)[-elite_count:]
    elite_points = points[elite_indices]
    elite_npvs = npvs[elite_indices]
    
    # Calculate weights
    if len(elite_npvs) > 1 and np.std(elite_npvs) > 1e-10:
        normalized_npvs = (elite_npvs - np.min(elite_npvs)) / (np.max(elite_npvs) - np.min(elite_npvs))
        weights = np.exp(exp_weight * normalized_npvs)
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(elite_npvs)) / len(elite_npvs)
    
    # Fit distribution
    mu = np.average(elite_points, axis=0, weights=weights)
    if is_univariate:
        var = np.average((elite_points[:, 0] - mu[0])**2, weights=weights)
        sigma = np.sqrt(max(var, min_variance))
        mu_sigma = {"mu": float(mu[0]), "sigma": float(sigma)}
    else:
        centered = elite_points - mu
        cov = np.zeros((2, 2))
        for i in range(len(elite_points)):
            cov += weights[i] * np.outer(centered[i], centered[i])
        
        # Regularize covariance
        cov += np.eye(2) * min_variance
        mu_sigma = {"mu": mu.tolist(), "sigma": cov.tolist()}
    
    # Generate new points
    new_points = set()    
    for _ in range(designs_per_run * 100):
        if len(new_points) >= designs_per_run:
            break
            
        # Sample from fitted distribution
        if is_univariate:
            sample = local_rng.normal(mu_sigma["mu"], mu_sigma["sigma"])
            sample = np.clip(sample, o2_range[0], o2_range[1])
            candidate = (round(sample, 2), 100.0)
        else:
            sample = local_rng.multivariate_normal(mu_sigma["mu"], mu_sigma["sigma"])
            sample[0] = np.clip(sample[0], o2_range[0], o2_range[1])
            sample[1] = np.clip(sample[1], comp_ratio_range[0], comp_ratio_range[1])
            candidate = (round(sample[0], 2), round(sample[1], 1))

        # Add if not already in all_designs or new_points
        candidate_key = f"{candidate[0]}__{candidate[1]}"
        if candidate_key not in all_designs and candidate not in new_points:
            new_points.add(candidate)
    
    return list(new_points), mu_sigma


def calculate_capacity_factor(month_data, upgrade_key, new_param_vals):
    """Calculate capacity factor based on storage type."""
    if "battery" in upgrade_key:
        storage_str = "E"
    elif "tank" in upgrade_key:
        storage_str = "N"

    if storage_str in month_data["profiles"] and month_data["profiles"][storage_str]:
        month_storage_profile = np.concatenate(month_data["profiles"][storage_str])
        month_storage_max = new_param_vals[f"{storage_str}_max"]
        return np.max(month_storage_profile) / month_storage_max
    return 0.0  # if no tank


def calculate_h2_metrics(month_data):
    """Calculate H2-related metrics."""
    if "Ndot_c" not in month_data["profiles"] or not month_data["profiles"]["Ndot_c"]:
        return {"H2": 0, "h2_value": 0}

    Ndot_c_profiles = np.concatenate(month_data["profiles"]["Ndot_c"])
    h2_moles = np.sum(Ndot_c_profiles) * 2  # 2 moles H2 per mole O2
    h2_kg = moles_to_mass(h2_moles, M_H2)
    h2_value = h2_kg * price_h2_kg
    return {"H2": h2_moles, "h2_value": h2_value}


def calculate_monthly_metrics(month_data, tariff_key, upgrade_key, new_param_vals):
    """Calculate monthly metrics including costs and energy metrics."""
    if not month_data["days"]:
        return {}, {}, {}
    
    # Get year, month explicitly from the first day
    year, month = map(int, month_data["days"][0].split("-")[:2])
    
    # Load tariff data and get charge dict for this month
    tariff_data = get_tariff_data()
    if tariff_key not in tariff_data:
        print(f"Warning: Tariff key {tariff_key} not found in tariff data")
        return {}, {}, {}
    
    charge_dict = get_charge_dict_for_month(tariff_data[tariff_key], year, month)

    # Check if we have any profiles
    if not month_data["profiles"]["Edot_t_net"]:
        return {}, {}, {}

    # Concatenate profiles from all days
    profiles = np.concatenate(month_data["profiles"]["Edot_t_net"])
    baseline_profiles = np.concatenate(month_data["profiles"]["Edot_t_baseline"])

    # Initialize results
    opex = {"new": {"h2": 0}, "baseline": {"h2": 0}, "savings": {"h2": 0}}

    new_itemized_costs, _ = costs.calculate_itemized_cost(
        charge_dict,
        {"electric": profiles},
        resolution="15m",
        desired_utility="electric",
        decompose_exports=True
    )

    baseline_itemized_costs, _ = costs.calculate_itemized_cost(
        charge_dict,
        {"electric": baseline_profiles},
        resolution="15m",
        desired_utility="electric",
        decompose_exports=True
    )

    # Extract individual costs
    for key in ("energy", "demand", "export", "customer"):
        opex["new"][key] = new_itemized_costs["electric"][key]
        opex["baseline"][key] = baseline_itemized_costs["electric"][key]
        opex["savings"][key] = opex["baseline"][key] - opex["new"][key]

    # Initialize metrics dictionary
    monthly_metrics = {
        "capacity_factor": calculate_capacity_factor(month_data, upgrade_key, new_param_vals),
        "month_rte": metrics.roundtrip_efficiency(baseline_profiles, profiles),
        "energy_capacity": metrics.energy_capacity(
            baseline_profiles,
            profiles,
            timestep=0.25,
            ec_type="discharging",
            relative=True,
        ),
        "month_power_capacity": metrics.power_capacity(
            baseline_profiles,
            profiles,
            timestep=0.25,
            pc_type="average",
            relative=True,
        ),
    }

    # Add H2 metrics if applicable
    if "elec" in upgrade_key:
        h2_metrics = calculate_h2_metrics(month_data)
        monthly_metrics["H2"] = h2_metrics["H2"]
        opex["new"]["h2"] = -h2_metrics["h2_value"]
        opex["savings"]["h2"] = h2_metrics["h2_value"]
    return {
        "monthly_metrics": monthly_metrics,
        "opex": opex
    }


def calculate_itemized_npv(electricity_savings_dict, capex, years=10):
    if np.isnan(capex):
        return {
            "by_component": {k: np.nan for k in electricity_savings_dict.keys()},
            "total": np.nan,
        }

    npv_dict = {"from capex": -capex, "total": -capex, "by_component": {}}
    for component, savings in electricity_savings_dict.items():
        component_value = metrics.net_present_value(
            capital_cost=0,
            electricity_savings=savings,
            maintenance_diff=0,
            timestep=0.25,
            upgrade_lifetime=years,
            )  # using default interest rate of 0.03
        npv_dict["by_component"][component] = component_value
        npv_dict["total"] += component_value

    return npv_dict


def get_max_dict_from_profile(profile, demand_charge_info_dict):
    """For each demand charge, get max from optimized profile for the window"""
    prev_demand_dict = {}
    
    if profile is None:  # Failed day
        return None
    
    power_profile = profile["Edot_t_net"]
    for charge_key, charge in demand_charge_info_dict.items():
        if "demand" not in charge_key:
            continue
        hour_start = charge["hour_start"]
        hour_end = charge["hour_end"]
        charge_rate = charge["charge_rate"]
        
        active_timesteps = [
            t for t in range(len(power_profile)) if hour_start <= t // 4 < hour_end
        ]
        if active_timesteps:
            max_demand = float(np.max(np.array(power_profile)[active_timesteps]))
            # Calculate the cost for this maximum demand
            max_cost = max_demand * charge_rate #TODO: use ecc function
            prev_demand_dict[charge_key] = {
                "demand": max_demand,  # Store the demand in kW
                "cost": max_cost       # Store the cost in $
            }
        else:
            prev_demand_dict[charge_key] = {"demand": 0.0, "cost": 0.0}
    return prev_demand_dict


def get_demand_charge_info_dict(tariff_key, month, year):
    """Get demand charge info dictionary for a given month"""
    # Load tariff data for this tariff_key
    tariff_data = get_tariff_data()
    if tariff_key not in tariff_data:
        print(f"Warning: Tariff key {tariff_key} not found in tariff data")
        return {}
    
    tariff_df = tariff_data[tariff_key]
    charge_dict = get_charge_dict_for_month(tariff_df, year, int(month))

    # Use CALENDAR_MONTHS to get the correct start (inclusive) and end (exclusive, first of next month)
    month_key = f"{year}-{int(month):02d}"
    if month_key not in CALENDAR_MONTHS:
        raise ValueError(f"Month key {month_key} not in CALENDAR_MONTHS")
    start_date = CALENDAR_MONTHS[month_key]["start"]
    end_date_exclusive = CALENDAR_MONTHS[month_key]["end"]  # first day of next month
    end_date_last_day = (np.datetime64(end_date_exclusive) - np.timedelta64(1, 'D')).astype(str)

    # Build demand_charge_info_dict with correct keys
    demand_charge_info_dict = {}
    for charge_key in charge_dict.keys():
        if "demand" not in str(charge_key):
            continue
        
        # Parse period from the charge key (e.g., 'electric_demand_peak-summer_20220701_20220801_0')
        parts = str(charge_key).split("_")
        charge_type = parts[0] + "_" + parts[1]  # e.g., 'electric_demand'
        period = parts[2]  # e.g., 'peak-summer'
        
        match = tariff_df[(tariff_df["type"] == "demand") & (tariff_df["period"] == period)]
        if match.empty:
            continue

        key = f"{charge_type}_{period}_{start_date.replace('-', '')}_{end_date_last_day.replace('-', '')}_0"
        demand_charge_info_dict[key] = {
            "charge_rate": match.iloc[0]["charge (metric)"],
            "hour_start": match.iloc[0]["hour_start"],
            "hour_end": match.iloc[0]["hour_end"],
        }
    return demand_charge_info_dict

def get_baseline_param_vals(Ndot_target, Edot_rem, base_wwtp_key):
    """
    Generate parameter values for a given scenario and o2_tech.

    Args:
        Ndot_target: Target oxygen flow rate
        Edot_rem: Remaining energy demand
        base_wwtp_key: Key specifying gas type and technology

    Returns:
        dict: A dictionary of parameter values
    """
    Ndot_target = copy.deepcopy(Ndot_target)

    P_b = P_AER + P_ATM
    gas, o2_tech_base, summer_multiplier, summer_smoothing = split_key(base_wwtp_key)
    P_init_base = P_init_map[o2_tech_base]
    ei_o2_base = ei_o2_map[o2_tech_base]
    ei_evap = ei_evap_kwh if o2_tech_base == "cryo" else 0
    Edot_b_comp_coeff = compressor_power_linear_coeff(P_init_base, P_b, rec=False)
    # print(Edot_b_comp_coeff)
    Edot_b_comp_baseline = Ndot_target / frac_o2_map[gas] * Edot_b_comp_coeff
    Edot_b_gen_baseline = Ndot_target * ei_o2_base # TODO: debug for EBMUD
    Edot_b_baseline = Edot_b_comp_baseline + Edot_b_gen_baseline
    return {
        "Ndot_target": Ndot_target,
        "Edot_rem": Edot_rem,
        "Edot_t_baseline": Edot_b_baseline + Edot_rem,
        "Edot_b_baseline": Edot_b_baseline,
        "P_init_base": P_init_base,
        "ei_o2_base": ei_o2_base,
        "ei_evap": ei_evap,
        "P_b": P_b,
        "Edot_b_comp_coeff": Edot_b_comp_coeff
    }


def get_remaining_new_param_vals_function(design_key, base_wwtp_key, upgrade_key, tariff_key, annual_param_limits):
    # Create problem instance just to get param vals once per design
    problem = O2Problem(
        single_day_config=f"{base_wwtp_key}___{upgrade_key}___{tariff_key}",
        design_key=design_key, date="dummy", data_wwtp='svcw',
    )
    param_vals = problem.get_remaining_new_param_vals(annual_param_limits)
    del problem  # clean up memory
    return param_vals
    
    
def run_configuration(config, run_name, designs_per_run, o2_range, comp_ratio_range,
                    n_jobs, max_iterations, skip_already_run=True, horizon_days=3, convergence_threshold = 0.005
):
    """Run optimization for a single configuration (combination of base facility, upgrade key, tariff key).
    
    Args:
        config: Configuration dictionary
        run_name: Base name for the run
        designs_per_run: Number of designs to run per iteration
        o2_range: Range of O2 storage hours
        comp_ratio_range: Range of compression ratios
        n_jobs: Number of parallel jobs
        max_iterations: Maximum number of optimization iterations
        skip_already_run: Whether to skip already completed runs
        horizon_days: Number of days to process in each rolling horizon (default 3
        convergence_threshold: improvement threshold to exit design search loop
    """
    output_dir = f"aeration_flexibility/output_data/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    base_wwtp_key = config["base_wwtp_key"]
    upgrade_key = config["upgrade_key"]
    tariff_key = config["tariff_key"]
    suffix = get_summer_key(config['summer_config']["multiplier"],config['summer_config']["smoothing"])
    config_name = get_config_name(base_wwtp_key, config['upgrade_key'], config['tariff_key'], suffix)

    if "battery" in upgrade_key:
        keys_to_append = ["Edot_t_baseline", "Edot_t_net", "E"]
    else:
        keys_to_append = ["Edot_t_baseline", "Edot_t_net", "N", "Ndot_c"]

    if skip_already_run and os.path.exists(f"aeration_flexibility/output_data/{run_name}/{config_name}.pkl"):
        print(f"Skipping {config_name}")
        return

    print(f"\nProcessing {config_name}")

    # Save intermediate file paths if available
    intermediate_dir = os.path.abspath(f"aeration_flexibility/output_data/{run_name}/intermediate/{config_name}")
    if not os.path.exists(intermediate_dir):
        print("   No intermediate files found")
        intermediate_filepaths = {}
    else:
        intermediate_filepaths = {}
        all_files = os.listdir(intermediate_dir)
        print(f"  Found {len(all_files)} files in directory")
        
        # Look for and store intermediate file paths
        for filename in all_files:
            if filename.endswith('.pkl'):  # structure design_key_month_key_date.pkl E.g.: 4.17__473.3_2022-07_2022-07-01.pkl
                base_name = filename.replace('.pkl', '')
                last_underscore_pos = base_name.rfind('_')
                second_last_underscore_pos = base_name.rfind('_', 0, last_underscore_pos)
                design_key = base_name[:second_last_underscore_pos]
                month_key = base_name[second_last_underscore_pos + 1:last_underscore_pos]
                date = base_name[last_underscore_pos + 1:]        
                if design_key not in intermediate_filepaths:
                    intermediate_filepaths[design_key] = {}
                if month_key not in intermediate_filepaths[design_key]:
                    intermediate_filepaths[design_key][month_key] = {}
                intermediate_filepaths[design_key][month_key][date] = os.path.abspath(os.path.join(intermediate_dir, filename))

    # Load existing results for this config
    all_results = {}
    config_file_path = os.path.join(output_dir, f"{config_name}.pkl")
    if os.path.exists(config_file_path):
        with open(config_file_path, 'rb') as f:
            config_results = pickle.load(f)
            all_results.update(config_results['all_results'])
            print(f"  Loaded existing results for {config_name}")

    # Load day profiles and group days by month
    with open(f"aeration_flexibility/output_data/{run_name}/day_profiles.pkl", "rb") as f:
        day_profiles = pickle.load(f)
    month_to_days = {}
    for date in sorted(day_profiles):
        year, month, _ = date.split("-")
        month_key = f"{year}-{int(month):02d}"
        if month_key not in month_to_days:
            month_to_days[month_key] = []
        month_to_days[month_key].append(date)

    gas, o2_tech_base, summer_multiplier, summer_smoothing = split_key(base_wwtp_key)

    # First loop to calculate baseline values based on Ndot_target
    baseline_val_dict = {}
    for date, day_data in day_profiles.items():
        year, month = day_data["year"], day_data["month"]
        month_key = f"{month:02d}"
        Ndot_target_day = (
            volume_to_moles_stp(day_data["profile"]["Blower_AerationBasin_Air_Flow"])
            * o2_multiplier_map[gas] * frac_o2_map['air'] / 24
        )  # Blower_AerationBasin_Air_Flow is in m3 air/day so divide by 24 and multiply by o2 fraction
        Ndot_target_day *= float(summer_multiplier)
        # if int(summer_smoothing) > 0:
        #     Ndot_target_day = np.convolve(Ndot_target_day, np.ones(int(summer_smoothing)) / int(summer_smoothing), mode="same")

        # predefine baseline param vals in order to calculate system annual_param_limits
        baseline_val_dict[date] = get_baseline_param_vals(
            Ndot_target=Ndot_target_day,
            Edot_rem=day_data["profile"]["VirtualDemand_RestOfFacilityPower"],
            base_wwtp_key=base_wwtp_key
        )

    # Calculate system annual_param_limits
    Ndot_target_year = np.array([val["Ndot_target"] for val in baseline_val_dict.values()])
    Edot_t_max = np.max(np.concatenate([val["Edot_t_baseline"] for val in baseline_val_dict.values()]))
    if float(summer_multiplier) > 1.0:
        Ndot_b_max = np.max(Ndot_target_year) / float(summer_multiplier)
        Edot_c_max = Edot_t_max / float(summer_multiplier)
        print(f"setting Ndot_b_max to {Ndot_b_max} (out of annual max {np.max(Ndot_target_year)} for summer_multiplier {summer_multiplier}")
    else:
        Ndot_b_max = np.max(Ndot_target_year)
        Edot_c_max = Edot_t_max
    
    annual_param_limits = {
        "Ndot_target_min": np.min(Ndot_target_year),
        "Ndot_target_max": np.max(Ndot_target_year),
        "Ndot_target_mean": np.mean(Ndot_target_year),
        "Ndot_b_max": Ndot_b_max,
        "Edot_t_max": Edot_t_max,
        "Edot_b_mean": np.mean([val["Edot_b_baseline"] for val in baseline_val_dict.values()]),
        "Edot_c_max": Edot_c_max,
    }

    # Initialize optimization
    if "design_point" in config:  # if specified (i.e for tornado)
        initial_design_points = [config["design_point"]]
    else:  # Specify initial design points
        if "battery" in upgrade_key or "liquid" in upgrade_key:
            h_min, h_max = o2_range
            h_center = (h_min + h_max) / 2
            
            initial_design_points = [
                # Fixed comparison point
                (1.0, 100.0),
                
                # Center and boundary points
                (h_center, 100.0),
                (h_min, 100.0),
                (h_max, 100.0),
                
                # Mid-points between center and boundaries
                ((h_min + h_center) / 2, 100.0),
                ((h_max + h_center) / 2, 100.0),
            ]
            initial_design_points = [(round(hours, 2), round(ratio, 1)) for hours, ratio in initial_design_points]
        else:
            # Calculate center and boundary points
            h_min, h_max = o2_range
            r_min, r_max = comp_ratio_range
            h_center = (h_min + h_max) / 2
            r_center = (r_min + r_max) / 2
            
            initial_design_points = [
                # Fixed comparison point
                (1.0, 100.0),
                
                # Center and corners
                (h_center, r_center),
                (h_min, r_min),  # bottom left
                (h_max, r_max),  # top right
                (h_max, r_min),  # bottom right
                (h_min, r_max),  # top left
                
                # Mid-points between center and corners
                ((h_min + h_center) / 2, (r_min + r_center) / 2),  # center to bottom-left
                ((h_max + h_center) / 2, (r_max + r_center) / 2),  # center to top-right
                ((h_max + h_center) / 2, (r_min + r_center) / 2),  # center to bottom-right
                ((h_min + h_center) / 2, (r_max + r_center) / 2),  # center to top-left
                
                # Edge centers (between center and edge midpoints)
                (h_center, r_min),  # bottom edge
                (h_center, r_max),  # top edge
                (h_min, r_center),  # left edge
                (h_max, r_center),  # right edge
            ]
            initial_design_points = [(round(hours, 2), round(ratio, 1)) for hours, ratio in initial_design_points]

    # Initialize results storage
    mu_sigma_history = []
    sampled_points_history = []
    iteration = 0
    infeasible_designs = set()

    # Initialize convergence tracking
    best_npv = -np.inf
    no_improvement_count = 0
    min_iterations = 3  # Minimum iterations before checking convergence

    # Get initial design points
    sampled_points_history.append(list(initial_design_points))

    # Main optimization loop
    while iteration < max_iterations:
        print(f"iteration {iteration}")

        # Get design points for this iteration
        if iteration == 0:
            design_points = initial_design_points
            mu_sigma_history.append(None)  # No mu/sigma for initial points
        else:
            # Sample new points using MVN
            design_points, mu_sigma = sample_from_multivariate_normal(all_results, designs_per_run, o2_range, comp_ratio_range, upgrade_key)
            if not design_points:
                print("No valid points generated for sampling. Stopping optimization.")
                break
            
            mu_sigma_history.append(mu_sigma)

        # Track which points were sampled this iteration
        sampled_points_history.append(list(design_points))
        unsolved_points = [
            pt for pt in design_points if f"{pt[0]}__{pt[1]}" not in all_results
        ]
        if not unsolved_points:
            print("All sampled design points solved. Next iteration.")
            iteration += 1
            continue

        # Get current best NPV and check for convergence
        current_best_npv = max(
            [
                result["metrics"]["npv"]["total"]
                for result in all_results.values()
                if is_valid_npv(result["metrics"]["npv"]["total"])
            ],
            default=-np.inf,
        )
        if iteration >= min_iterations and current_best_npv > best_npv:
            improvement = (
                (current_best_npv - best_npv) / abs(best_npv)
                if best_npv != -np.inf
                else np.inf
            )
            if improvement < convergence_threshold:
                no_improvement_count += 1
                if no_improvement_count >= 2:
                    print(f"Convergence reached after {iteration + 1} iterations")
                    print(f"Best NPV: ${current_best_npv:,.2f}")
                    break
            else:
                no_improvement_count = 0
                print(f"Current best NPV: ${current_best_npv:,.2f}, continuing")
            best_npv = current_best_npv
        # Sort design points by hours (descending) and create job queue with design point-month combinations
        sorted_unsolved_points = sorted(unsolved_points, key=lambda x: x[0], reverse=True)
        print(f"  Creating job queue from {len(sorted_unsolved_points)} unsolved points")
        print(f"  Current infeasible_designs: {sorted(list(infeasible_designs))}")
        
        # First pass: check intermediate files and mark designs as infeasible
        for Hours_of_O2, compression_ratio in sorted_unsolved_points:
            design_key = f"{Hours_of_O2}__{compression_ratio}"
            if intermediate_filepaths and design_key in intermediate_filepaths:
                for month_key, month_files in intermediate_filepaths[design_key].items():
                    failed_days = 0
                    total_days = len(month_files)
                    for day_file in month_files.values():
                        with open(day_file, 'rb') as f:
                            day_data = pickle.load(f)
                            if not day_data.get("profile") or "Edot_t_net" not in day_data.get("profile", {}):
                                failed_days += 1
                    if failed_days > total_days / 2:
                        infeasible_designs.add(design_key)
                        print(f"  Marking {design_key} as infeasible based on intermediate files ({failed_days}/{total_days} failed days in {month_key})")
                        break
        
        # Second pass: apply hierarchical pruning - if a larger design fails, all smaller designs are infeasible
        if infeasible_designs:
            max_failed_hours = max(float(design.split('__')[0]) for design in infeasible_designs)            
            for Hours_of_O2, compression_ratio in sorted_unsolved_points:
                design_key = f"{Hours_of_O2}__{compression_ratio}"
                if float(Hours_of_O2) <= max_failed_hours and design_key not in infeasible_designs:
                    infeasible_designs.add(design_key)
                    print(f"  Marking {design_key} as infeasible (≤{max_failed_hours} hours)")
        
        # Create job queue, skipping infeasible designs
        job_queue = []
        skipped_count = 0
        for Hours_of_O2, compression_ratio in sorted_unsolved_points:
            design_key = f"{Hours_of_O2}__{compression_ratio}"
            
            # Skip this design if already known to be infeasible
            if design_key in infeasible_designs:
                skipped_count += 1
                continue
                
            new_param_vals = get_remaining_new_param_vals_function(
                design_key, base_wwtp_key, upgrade_key, tariff_key, annual_param_limits
                )

            for month_key, month_days in month_to_days.items():
                job_queue.append((design_key, month_key, month_days, base_wwtp_key, upgrade_key, tariff_key,
                                  None, baseline_val_dict, new_param_vals))
        print(f"  Created {len(job_queue)} jobs, skipped {skipped_count} infeasible designs")

        # Create parent directories for intermediate results
        for Hours_of_O2, compression_ratio in sorted_unsolved_points:
            design_key = f"{Hours_of_O2}__{compression_ratio}"
            intermediate_dir = f"aeration_flexibility/output_data/{run_name}/intermediate/{config_name}"
            os.makedirs(intermediate_dir, exist_ok=True)

        # Track infeasible design points to prune smaller hours
        results = []

        # Process jobs with dynamic pruning
        print(f"Processing {len(job_queue)} jobs with {n_jobs} workers...")
        with Pool(processes=n_jobs) as pool:
            # Submit initial batch of up to n_jobs jobs
            pending_jobs = {}
            job_index = 0
            while len(pending_jobs) < n_jobs and job_index < len(job_queue):
                job_data = job_queue[job_index]
                design_key = job_data[0]  # design_key is now the first element
                
                # Skip if this design is already known to be infeasible
                if design_key in infeasible_designs:
                    job_index += 1
                    continue
                
                # Submit job
                async_result = pool.apply_async(process_design_point_and_month, 
                                                args=(job_data, run_name, intermediate_filepaths, horizon_days))
                pending_jobs[async_result] = (job_index, design_key)
                job_index += 1
            
            # Process completed jobs and submit new ones
            while pending_jobs:
                # Wait for any job to complete
                for async_result in list(pending_jobs.keys()):
                    if async_result.ready():
                        job_idx, design_key = pending_jobs.pop(async_result)
                        result = async_result.get()
                        results.append(result)
                        
                        if result and len(result) >= 6:
                            month_key = result[1]
                            month_results = result[2]  # month_results
                            new_param_vals = result[3]
                            is_feasible = result[4]  # is_feasible
                            failure_reason = result[5]  # failure_reason

                            if not is_feasible:
                                infeasible_designs.add(design_key)
                                print(f"Design {design_key} marked as {failure_reason}, pruning smaller hours")
                        
                        # Submit next job if available
                        while job_index < len(job_queue):
                            job_data = job_queue[job_index]
                            design_key = job_data[0]  # design_key is now the first element
                            
                            # Skip if this design is already known to be infeasible
                            if design_key in infeasible_designs:
                                job_index += 1
                                continue
                            
                            async_result = pool.apply_async(process_design_point_and_month, 
                                                            args=(job_data, run_name, intermediate_filepaths, horizon_days))
                            pending_jobs[async_result] = (job_index, design_key)
                            job_index += 1
                            break

        # Process results and aggregate by design point
        for Hours_of_O2, compression_ratio in unsolved_points:
            design_key = f"{Hours_of_O2}__{compression_ratio}"
            design_results = [r for r in results if r[0] == design_key]

            if not design_results:
                continue

            # Aggregate results for this design point
            monthly_data = {}
            monthly_max_values = {}
            month_failures = {}
            design_failed = False  # Flag to track if design should be skipped

            for (
                design_key_from_result,
                month_key,
                month_results,
                new_param_vals,
                is_feasible,
                failure_reason,
            ) in design_results:
                # Check if this month indicates design failure
                if not is_feasible:
                    print(f"  ✗ Design {design_key} failed in {month_key}: {failure_reason}")
                    all_results[design_key] = {
                        "design_key": design_key,
                        "metrics": {
                            "npv": {
                                "total": -np.inf,
                                "by_component": {"energy": -np.inf, "demand": -np.inf, "export": -np.inf, "h2": -np.inf
                                }
                            },
                            "capex": np.inf,
                            "tank_metrics": None,
                        },
                        "new_param_vals": new_param_vals,
                        "infeasibility_reason": failure_reason,
                    }
                    design_failed = True
                    break  # Exit month loop
                    
                monthly_max_values[month_key] = {"Ndot_c": 0, "Edot_c": 0}

                # Initialize monthly_data structure for this month with only key profiles
                monthly_data[month_key] = {
                    "days": [],
                    "profiles": {
                        "Edot_t_net": [], "Edot_t_baseline": [],
                        "E": [], "N": [], "Ndot_c": [],
                    },
                }

                # Process each day's results and add only key profiles to monthly_data
                for date, (profile, max_values) in month_results.items():
                    if profile and "Edot_t_net" in profile:
                        monthly_data[month_key]["days"].append(date)
                        for key in ["Ndot_c", "Edot_c"]:
                            monthly_max_values[month_key][key] = max(monthly_max_values[month_key][key], max_values[key])
                        for key in keys_to_append:
                            monthly_data[month_key]["profiles"][key].append(profile[key])
                
                # Handle replacements for missing / unsolved days
                if monthly_data[month_key]["days"]:
                    year, month = map(int, monthly_data[month_key]["days"][0].split("-")[:2])
                    all_days = get_all_days_in_month(year, month)
                    valid_days = monthly_data[month_key]["days"]
                    for missing_date in set(all_days) - set(valid_days):
                        replacement_days = get_replacement_days(date)
                        for replacement_day in replacement_days:
                            if replacement_day in valid_days:
                                print(f"Using {replacement_day} as replacement for {missing_date}")
                                monthly_data[month_key]["days"].append(missing_date)
                                for key_to_append in keys_to_append:
                                    monthly_data[month_key]["profiles"][key_to_append].append(
                                        monthly_data[month_key]["profiles"][key_to_append][valid_days.index(replacement_day)]
                                    )
                    
                    # Sort days and profiles to ensure correct order
                    sorted_indices = np.argsort(monthly_data[month_key]["days"])
                    for key in monthly_data[month_key]["profiles"]:
                        if len(monthly_data[month_key]["profiles"][key]) == len(sorted_indices):
                            monthly_data[month_key]["profiles"][key] = [
                                monthly_data[month_key]["profiles"][key][i] for i in sorted_indices
                            ]
                    monthly_data[month_key]["days"] = [monthly_data[month_key]["days"][i] for i in sorted_indices]
            
            # Calculate metrics
            if design_failed:
                print(f"  Skipping metrics calculation for {design_key} - design marked as failed")
                continue
                
            opex_keys = ["h2", "energy", "demand", "export", "customer"]
            annual_opex = {
                name: {k: 0 for k in opex_keys}
                for name in ["new", "baseline", "savings"]
            }
            metric_lists = {
                k: []
                for k in [
                    "month_rte",
                    "energy_capacity",
                    "month_power_capacity",
                    "capacity_factor",
                ]
            }

            for month_key, month_data in monthly_data.items():
                complete_monthly_data = calculate_monthly_metrics(
                    month_data, tariff_key, upgrade_key, new_param_vals
                )
                if not complete_monthly_data["opex"]:
                    print(f"  Skipping metrics calculation for {month_key} due to empty monthly_opex")
                    continue
                for name in ["new", "baseline", "savings"]:
                    for k in opex_keys:
                        annual_opex[name][k] += complete_monthly_data["opex"][name][k]
                for k in metric_lists:
                    metric_lists[k].append(complete_monthly_data["monthly_metrics"][k])

            # Scale annual OPEX values to 1-year for annual savings
            num_months = len(monthly_data)
            if num_months != 12:
                scale_factor = 12 / num_months
                for name in ["new", "baseline", "savings"]:
                    for k in opex_keys:
                        annual_opex[name][k] *= scale_factor

            # Calculate final metrics
            annual_metrics = {
                "energy_metrics": {
                    "rte": np.mean(metric_lists["month_rte"]),
                    "mean_energy_capacity": np.mean(metric_lists["energy_capacity"]),
                    "mean_power_capacity": np.mean(metric_lists["month_power_capacity"]),
                    "capacity_factor": np.mean(metric_lists["capacity_factor"]),
                },
                "opex": annual_opex,
            }


            # Calculate maximum values across all months for capex calculation
            max_ndot_c = max(
                monthly_max_values[mk]["Ndot_c"] 
                for mk in monthly_max_values.keys()
            ) if monthly_max_values else 0
            max_edot_c = max(
                monthly_max_values[mk]["Edot_c"] 
                for mk in monthly_max_values.keys()
            ) if monthly_max_values else 0
            
            capex, tank_metrics, capex_components, counterfactual_capex = (
                calculate_capex(
                    sub_dict={
                        "param_vals": {
                            "V_tank": new_param_vals.get("V_tank", 0),
                            "N_max": new_param_vals.get("N_max", 0),
                            "E_max": new_param_vals.get("E_max", 0),
                            "P_max": new_param_vals.get("P_max", 0),
                            "solar_multiplier": new_param_vals["solar_multiplier"],
                            "max_Edot_c": max_edot_c,
                        },
                        "max_values": {
                            "Ndot_c": max_ndot_c,
                            "Edot_c": max_edot_c,
                        },
                        "billing_key": tariff_key,
                    },
                    storage_type=upgrade_key.split("__")[1],
                    new_o2_supply_tech=upgrade_key.split("__")[0],
                    base_wwtp_key=base_wwtp_key,
                    new_param_vals=new_param_vals,
                )
            )

            # Calculate NPV
            annual_savings = {
                "energy": annual_opex["savings"]["energy"],
                "demand": annual_opex["savings"]["demand"],
                "export": annual_opex["savings"]["export"],
                "h2": annual_opex["savings"]["h2"],
            }
            npv = calculate_itemized_npv(annual_savings, capex)
            npv["from capex savings"] = counterfactual_capex
            print(
                f" {design_key} NPV energy ${npv['by_component']['energy']:,.2f} demand ${npv['by_component']['demand']:,.2f} export ${npv['by_component']['export']:,.2f} H2 ${npv['by_component']['h2']:,.2f} Total ${npv['total']:,.2f} capex {capex}"
            )

            # Store results
            annual_metrics.update(
                {"npv": npv, "capex": capex, "tank_metrics": tank_metrics}
            )
            all_results[design_key] = {
                "design_key": design_key,
                "metrics": annual_metrics,
                "month_profiles": monthly_data,
                "new_param_vals": new_param_vals,
            }

            # Clean up memory
            del monthly_data
            del monthly_max_values
            del month_failures
            del annual_metrics
            del annual_opex
            del metric_lists
            del annual_savings
            del capex_components

        del results
        gc.collect()

        iteration += 1

    # Save results
    results_data = {
        "all_results": all_results,
        "mu_sigma_history": mu_sigma_history,
        "sampled_points_history": sampled_points_history,
    }
    with open(f"aeration_flexibility/output_data/{run_name}/{config_name}.pkl", "wb") as f:
        pickle.dump(results_data, f)
    return results_data
