import numpy as np
import pickle
import pandas as pd
import os
import gc
import time
from multiprocessing import Pool

from electric_emission_cost import costs, metrics

from helpers.operational_optimization_problem import *
from helpers.capex_npv_energy_metrics import *
from helpers.parameters import *
from helpers.config_labels import *
from helpers.tariffs import get_tariff_data, get_charge_dict_for_month, CALENDAR_MONTHS, get_all_days_in_month, get_replacement_days

# Set numpy random seed to match tariffs.py
np.random.seed(RANDOM_SEED)

def concatenate_n_days_data(baseline_val_dict, charge_dict_month, group_days):
    """
    Concatenate data for N consecutive days starting from start_day_idx.
    
    Args:
        baseline_val_dict: Dictionary of baseline values for each day
        charge_dict_month: Monthly charge dictionary
        group_days: Days for processing
    
    Returns:
        tuple: (concatenated_baseline_vals, concatenated_charge_dict)
    """
    
    # Concatenate indexed baseline values
    concatenated_baseline_vals = {}
    for key in baseline_val_dict[group_days[0]].keys():
        if isinstance(baseline_val_dict[group_days[0]][key], np.ndarray):
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
    
    return concatenated_baseline_vals, concatenated_charge_dict


def process_n_days(group_days, design_key, base_wwtp_key, upgrade_key, tariff_key, new_param_vals, 
                  baseline_val_dict, charge_dict_month, prev_demand_dict=None,
                  initial_SoS=None, horizon_days=3):
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
        initial_SoS: Initial storage state for the first day
    
    Returns:
        tuple: (profiles_dict, new_param_vals, max_values_dict, initial_SoSs_dict)
    """
    
    concatenated_baseline_vals, concatenated_charge_dict = concatenate_n_days_data(
        baseline_val_dict, charge_dict_month, group_days
    )

    # Create problem instance for N days
    problem_start = time.time()
    var_length = len(concatenated_baseline_vals["Ndot_target"])
    problem = O2Problem(
        prob_name=get_config_name(base_wwtp_key, upgrade_key, tariff_key),
        design_key=design_key, date=f"{group_days[0]}_to_{group_days[-1]}", 
        initial_SoS=initial_SoS, horizon_days=horizon_days, var_length=var_length
    )

    # Check if concatenated data has correct length
    if var_length != len(group_days) * 96:
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
    print(f"  Problem Solve time: {(time.time() - solve_start):.3f}s, Total: {(time.time() - problem_start):.3f}s")
    
    # problem.print_cost_values(charge_dict=concatenated_charge_dict, prev_demand_dict=prev_demand_dict)
    del problem.m
    del problem
    
    if profile and "Edot_t_net" in profile:
        profiles_dict = {}
        max_values_dict = {}
        
        for day_idx, day in enumerate(group_days):

            # Extract day-specific profile
            day_profile = {}
            start_idx = day_idx * 96
            for key, value in profile.items():
                # Check if this is time-series data
                if hasattr(value, '__len__') and len(value) == len(group_days) * 96:
                    day_profile[key] = pd.Series(value[start_idx:start_idx + 96])
                else:  # non-time-series data
                    day_profile[key] = value
            
            if day_profile and "Edot_t_net" in day_profile:
                profiles_dict[day] = day_profile
                
                # Calculate max values for this day
                max_values_dict[day] = {
                    "Ndot_c": np.max(day_profile["Ndot_c"]) if "Ndot_c" in day_profile else 0,
                    "Edot_c": np.max(day_profile["Edot_c"]) if "Edot_c" in day_profile else 0,
                }
                
                # Store initial storage state for the second to last day of the group
                if day_idx == len(group_days) - 2 or len(group_days)<horizon_days:  # Second to last day
                    initial_SoS_next_run = extract_final_storage_state(day_profile)
            else:
                profiles_dict[day] = None
                max_values_dict[day] = {"Ndot_c": 0, "Edot_c": 0}
                initial_SoS_next_run = None
        
        return profiles_dict, max_values_dict, initial_SoS_next_run

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
    
    failed_days = sum(1 for profile, _ in month_results.values() 
                     if not profile or "Edot_t_net" not in profile)
    
    # Check for insufficient storage
    if failed_days >= 10:
        if float(summer_multiplier) > 1.0:
            return False, "insufficient_storage", failed_days
        else:
            return False, "infeasible", failed_days
    
    return True, None, failed_days


def skip_saving_last_day(horizon_days, i, day_idx, days):
    is_last_day_of_month = (i + horizon_days >= len(days))
    is_last_day_of_horizon = (day_idx == horizon_days - 1)
    return is_last_day_of_horizon and not is_last_day_of_month


def process_design_point_and_month(design_point_data, run_name, tariff_data, intermediate_filepaths=None, horizon_days=3):
    """Process a single design point and month combination using N-day optimization with 1-day overlap.

    Args:
        design_point_data: Tuple of (design_key, month_key, month_days, base_wwtp_key, upgrade_key,
                                    tariff_key, new_param_vals, baseline_val_dict, new_param_vals, annual_param_limits)
        run_name: Name of the run for saving intermediate results
        intermediate_filepaths: Dictionary of file paths to already-solved intermediate results
        horizon_days: Number of days to process in each rolling horizon (default 3)
    """
    (design_key, month_key, month_days, base_wwtp_key, upgrade_key,
     tariff_key, new_param_vals, baseline_val_dict, new_param_vals, annual_param_limits) = design_point_data

    # Process the month using N-day optimization with 1-day overlap
    month_start_time = time.time()
    year, month = month_key.split("-")
    
    print(f"processing {design_key} in month {month}, {year}")

    # Initialize prev_demand_dict with zero demands for all charge types
    prev_demand_dict = {}
    demand_charge_info_dict = get_demand_charge_info_dict(
        tariff_key, month, int(year), tariff_data
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
                print(f"  {design_key} Found dummy files for days: {dummy_file_days}")
        else:
            initial_SoS = None
            if i == 0:
                if float(summer_multiplier) > 1.0: # Give 1 hr storage on first day for feasibility
                    # Set initial storage to 1 hour of Ndot_target_mean for even playing field
                    initial_SoS = {'N': annual_param_limits["Ndot_target_mean"], 'E': 0}
                else:  # O2Problem will use minimum values
                    pass
            elif i > 0:
                # Use initial storage state from the last day of the previous group
                prev_group_last_day = month_days[i-1]  # This was the last day in the previous group
                if prev_group_last_day in month_results:
                    prev_profile = month_results[prev_group_last_day][0]
                    if prev_profile:
                        # Get the initial storage state (first timestep) of the last day from previous group
                        initial_SoS = extract_final_storage_state(prev_profile)
            
            # Process the N-day group
            profiles_dict, max_values_dict, final_storage_state_prev_run = process_n_days(
                group_days, design_key, base_wwtp_key, upgrade_key, tariff_key, new_param_vals,
                baseline_val_dict, charge_dict_month, prev_demand_dict=prev_demand_dict,
                initial_SoS=initial_SoS, horizon_days=horizon_days
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
                        "initial_SoS": final_storage_state_prev_run,
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
                        "initial_SoS": initial_SoS,  # Preserve the storage state from previous horizon
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
                return (month_key, month_results, new_param_vals, False, failure_reason)
        
        i += (horizon_days - 1)  # include 1 day overlap for next group

    is_feasible, failure_reason, _ = check_design_feasibility(month_results, summer_multiplier)

    print(f"Month {month_key} for design key {design_key} completed in {time.time() - month_start_time:.3f}s")

    return {
        "month_key": month_key,
        "month_results": month_results,
        "new_param_vals": new_param_vals,
        "is_feasible": is_feasible,
        "failure_reason": failure_reason,
    }


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
        print('No valid points generated for sampling. Stopping optimization.')
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



def calculate_metrics_from_annual_profiles(annual_profiles, tariff_key, upgrade_key, new_param_vals, tariff_data, num_months=12, start_date=None, end_date=None):
    """Calculate annual metrics from concatenated profiles."""
    if not annual_profiles["Edot_t_net"]:
        return {"energy_metrics": {}, "opex": {}}
    
    profiles = np.concatenate(annual_profiles["Edot_t_net"])
    baseline_profiles = np.concatenate(annual_profiles["Edot_t_baseline"])
    
    # Initialize results
    opex = {"new": {"h2": 0}, "baseline": {"h2": 0}, "savings": {"h2": 0}}

    charge_dict = costs.get_charge_dict(
        start_date, end_date, tariff_data[tariff_key], resolution="15m"
    )
    
    # Recalculate costs
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
    
    # Scale annual OPEX values to 1-year for annual savings
    if num_months != 12:
        scale_factor = 12 / num_months if num_months > 0 else 0
        for name in ["new", "baseline", "savings"]:
            for k in ["energy", "demand", "export", "customer"]:
                opex[name][k] *= scale_factor
    
    # Initialize metrics dictionary
    store_id = "E" if "battery" in upgrade_key else "N"
    energy_metrics = {
        "capacity_factor": np.max(np.concatenate(annual_profiles[store_id])) / new_param_vals[f"{store_id}_max"],
        "rte": metrics.roundtrip_efficiency(baseline_profiles, profiles),
        "mean_energy_capacity": metrics.energy_capacity(
            baseline_profiles,
            profiles,
            timestep=0.25,
            ec_type="discharging",
            relative=True,
        ),
        "mean_power_capacity": metrics.power_capacity(
            baseline_profiles,
            profiles,
            timestep=0.25,
            pc_type="average",
            relative=True,
        ),
    }
    
    # Add H2 to metrics if applicable
    if "elec" in upgrade_key:
        Ndot_c_profiles = np.concatenate(annual_profiles["Ndot_c"])
        h2_moles = np.sum(Ndot_c_profiles) * 2  # 2 moles H2 per mole O2
        h2_value = moles_to_mass(h2_moles, M_H2) * price_h2_kg
        energy_metrics["H2"] = h2_moles
        opex["new"]["h2"] = -h2_value
        opex["savings"]["h2"] = h2_value
    
    return {
        "energy_metrics": energy_metrics,
        "opex": opex
    }


def calculate_itemized_npv(annual_metrics, capex, years=10):
    """Calculate NPV from consolidated annual metrics."""
    if np.isnan(capex):
        return {"by_component": {}, "total": np.nan}
    
    npv_dict = {"from capex": -capex, "total": -capex, "by_component": {}}
    savings = annual_metrics.get("opex", {}).get("savings", {})    
    for component, amount in savings.items():
        if component in ["energy", "demand", "export", "h2"]:
            component_npv = metrics.net_present_value(
                capital_cost=0,
                electricity_savings=amount,
                maintenance_diff=0,
                timestep=0.25,
                upgrade_lifetime=years
            )
            npv_dict["by_component"][component] = component_npv
            npv_dict["total"] += component_npv
    
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


def get_demand_charge_info_dict(tariff_key, month, year, tariff_data):
    """Get demand charge info dictionary for a given month"""
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


def get_baseline_param_vals(day_data, base_wwtp_key):
    """
    Generate baseline parameter values for a given day and o2_tech.

    Args:
        day_data: 
        base_wwtp_key: Key specifying gas type and technology

    Returns:
        dict: A dictionary of parameter values
    """

    gas, o2_tech_base, summer_multiplier, summer_smoothing = split_key(base_wwtp_key)

    year, month = day_data["year"], day_data["month"]
    Ndot_target_day = (
        volume_to_moles_stp(day_data["profile"]["Blower_AerationBasin_Air_Flow"])
        * o2_multiplier_map[gas] * frac_o2_map['air'] / 24
    )  # Blower_AerationBasin_Air_Flow is in m3 air/day so divide by 24 and multiply by o2 fraction
    Ndot_target_day *= float(summer_multiplier)
    
    # if int(summer_smoothing) > 0:
    #     Ndot_target_day = np.convolve(Ndot_target_day, np.ones(int(summer_smoothing)) / int(summer_smoothing), mode="same")


    P_b = P_AER + P_ATM
    P_init_base = P_init_map[o2_tech_base]
    ei_o2_base = ei_o2_map[o2_tech_base]
    ei_evap = ei_evap_kwh if o2_tech_base == "cryo" else 0
    
    Edot_b_comp_coeff = compressor_power_linear_coeff(P_init_base, P_b, rec=False)
    # print(Edot_b_comp_coeff)

    Edot_b_comp_baseline = Ndot_target_day / frac_o2_map[gas] * Edot_b_comp_coeff
    Edot_b_gen_baseline = Ndot_target_day * ei_o2_base # TODO: debug for EBMUD
    Edot_b_baseline = Edot_b_comp_baseline + Edot_b_gen_baseline

    Edot_rem = day_data["profile"]["VirtualDemand_RestOfFacilityPower"]

    return {
        "Ndot_target": Ndot_target_day,
        "Edot_rem": Edot_rem,
        "Edot_t_baseline": Edot_b_baseline + Edot_rem,
        "Edot_b_baseline": Edot_b_baseline,
        "P_init_base": P_init_base,
        "ei_o2_base": ei_o2_base,
        "ei_evap": ei_evap,
        "P_b": P_b,
        "Edot_b_comp_coeff": Edot_b_comp_coeff
    }
   

def get_initial_design_points(config, o2_range, comp_ratio_range, upgrade_key):
    """Generate initial design points for optimization."""
    if "design_point" in config:  # if specified (i.e for tornado)
        return [config["design_point"]]
    
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
        return [(round(hours, 2), round(ratio, 1)) for hours, ratio in initial_design_points]
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
        return [(round(hours, 2), round(ratio, 1)) for hours, ratio in initial_design_points]


def submit_job(pool, job_queue, run_name, tariff_data, intermediate_filepaths, horizon_days, pending_jobs, job_index, infeasible_designs):
    """Submit a job to the multiprocessing pool."""
    job_data = job_queue[job_index]
    design_key = job_data[0]
    
    # Skip if this design is already known to be infeasible
    if design_key in infeasible_designs:
        return job_index + 1
    
    async_result = pool.apply_async(process_design_point_and_month, 
                                    args=(job_data, run_name, tariff_data, intermediate_filepaths, horizon_days))
    pending_jobs[async_result] = (job_index, design_key)
    return job_index + 1


def run_configuration(config, run_name, designs_per_run, o2_range, comp_ratio_range,
                    n_jobs, max_iterations, skip_already_run=True, horizon_days=3, convergence_threshold = 0.005):
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

    tariff_data = get_tariff_data()
    gas, o2_tech_base, summer_multiplier, summer_smoothing = split_key(base_wwtp_key)

    # Load day profiles and group days by month
    with open(f"aeration_flexibility/output_data/{run_name}/day_profiles.pkl", "rb") as f:
        day_profiles = pickle.load(f)

    # First loop through days to calculate baseline operation parameter values Ndot_target and Energy
    month_to_days = {}
    baseline_val_dict = {}
    for date, day_data in sorted(day_profiles.items()):

        # Map month to days
        year, month, _ = date.split("-")
        month_key = f"{year}-{int(month):02d}"
        if month_key not in month_to_days:
            month_to_days[month_key] = []
        month_to_days[month_key].append(date)

        # Get baseline val dict
        baseline_val_dict[date] = get_baseline_param_vals(day_data=day_data, base_wwtp_key=base_wwtp_key)

    # Calculate system annual_param_limits
    Ndot_target_year = np.array([val["Ndot_target"] for val in baseline_val_dict.values()])
    Edot_t_baseline_max = np.max(np.concatenate([val["Edot_t_baseline"] for val in baseline_val_dict.values()]))
    if float(summer_multiplier) > 1.0:
        Ndot_b_max = np.max(Ndot_target_year) / float(summer_multiplier)
        Edot_c_max = Edot_t_baseline_max * 2
        print(f"setting Ndot_b_max to {Ndot_b_max} (out of annual max {np.max(Ndot_target_year)} for summer_multiplier {summer_multiplier}")
    else:
        Ndot_b_max = np.max(Ndot_target_year)
        Edot_c_max = Edot_t_baseline_max
    
    annual_param_limits = {
        "Ndot_target_min": np.min(Ndot_target_year),
        "Ndot_target_max": np.max(Ndot_target_year),
        "Ndot_target_mean": np.mean(Ndot_target_year),
        "Ndot_b_max": Ndot_b_max,
        "Edot_t_max": Edot_t_baseline_max * 2,
        "Edot_b_mean": np.mean([val["Edot_b_baseline"] for val in baseline_val_dict.values()]),
        "Edot_c_max": Edot_c_max,
    }

    # Initialize optimization
    if "design_point" in config:  # if specified (i.e for tornado)
        initial_design_points = [config["design_point"]]
    else:  # Specify initial design points
        initial_design_points = get_initial_design_points(config, o2_range, comp_ratio_range, upgrade_key)

    # Initialize results storage and convergence tracking
    mu_sigma_history = [None]
    sampled_points_history = []
    sampled_points_history.append(list(initial_design_points))
    iteration = 0
    infeasible_designs = set()
    best_npv = -np.inf  # saving highest NPV so far
    no_improvement_count = 0  # number of iterations with no improvement
    min_iterations = 3  # Minimum iterations before checking convergence
    
    # Main optimization loop
    while iteration < max_iterations:
        print(f"iteration {iteration}")

        # GET DESIGN POINTS TO SOLVE FOR THIS ITERATION
        if iteration == 0:
            design_points = initial_design_points
        else:  # Sample new points using MVN
            design_points, mu_sigma = sample_from_multivariate_normal(all_results, designs_per_run, o2_range, comp_ratio_range, upgrade_key)
            if not design_points:
                break
            mu_sigma_history.append(mu_sigma)
        sampled_points_history.append(list(design_points))
        unsolved_points = [pt for pt in design_points if f"{pt[0]}__{pt[1]}" not in all_results]
        
        if not unsolved_points:
            iteration += 1
            continue

        # GET CURRENT BEST NPV AND CHECK FOR CONVERGENCE
        current_best_npv = max(
            [
                result["metrics"]["npv"]["total"]
                for result in all_results.values()
                if is_valid_npv(result["metrics"]["npv"]["total"])
            ],
            default=-np.inf,
        )
        if iteration >= min_iterations and current_best_npv > best_npv:
            if best_npv != -np.inf:
                improvement = (current_best_npv - best_npv) / abs(best_npv)
            else:
                improvement = np.inf

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

        # CREATE JOB QUEUE WITH DESIGN - MONTH COMBINATIONS
        sorted_unsolved_points = sorted(unsolved_points, key=lambda x: x[0], reverse=True)  # solve largest to smallest
        print(f"  Creating job queue from {len(sorted_unsolved_points)} unsolved points")
        print(f"  Current infeasible_designs: {sorted(list(infeasible_designs))}")
        # Check intermediate files and mark designs as infeasible if already assessed
        job_queue = []
        skipped_count = 0
        for Hours_of_O2, compression_ratio in sorted_unsolved_points:
            design_key = f"{Hours_of_O2}__{compression_ratio}"
            if intermediate_filepaths and design_key in intermediate_filepaths:
                failed_days = 0
                for month_key, month_files in intermediate_filepaths[design_key].items():
                    for day_file in month_files.values():
                        with open(day_file, 'rb') as f:
                            day_data = pickle.load(f)
                        if not day_data.get("profile") or "Edot_t_net" not in day_data.get("profile", {}):
                            failed_days += 1
                if failed_days >= 10:
                    infeasible_designs.add(design_key)
                    print(f"  Marking {design_key} as infeasible based on intermediate files w/ {failed_days} failed days")
            
            # Apply hierarchical pruning - if a larger design fails, all smaller designs are infeasible
            if infeasible_designs:
                max_failed_hours = max(float(design.split('__')[0]) for design in infeasible_designs)            
                if float(Hours_of_O2) <= max_failed_hours and design_key not in infeasible_designs:
                    infeasible_designs.add(design_key)
                    print(f"  Marking {design_key} as infeasible (≤{max_failed_hours} hours)")
            design_key = f"{Hours_of_O2}__{compression_ratio}"
            if design_key in infeasible_designs:
                skipped_count += 1
                continue

            # Create problem instance just to get param vals once per design
            problem = O2Problem(
                prob_name=get_config_name(base_wwtp_key, upgrade_key, tariff_key),
                design_key=design_key
            )
            new_param_vals = problem.get_remaining_new_param_vals(annual_param_limits)
            del problem

            # Create job
            for month_key, month_days in month_to_days.items():
                job_queue.append((design_key, month_key, month_days, base_wwtp_key, upgrade_key, tariff_key,
                                  None, baseline_val_dict, new_param_vals, annual_param_limits))
            
            # Create parent directory for intermediate results
            intermediate_dir = f"aeration_flexibility/output_data/{run_name}/intermediate/{config_name}"
            os.makedirs(intermediate_dir, exist_ok=True)
            
        # PROCESS JOBS, TRACKING RESULTS AND DYNAMICALLY REMOVING FURTHER INFEASIBLE DESIGNS
        print(f"Processing {len(job_queue)} jobs with {n_jobs} workers, skipped {skipped_count} infeasible designs")
        results = []
        with Pool(processes=n_jobs) as pool:
            # Submit initial batch of up to n_jobs jobs
            pending_jobs = {}
            job_index = 0
            while len(pending_jobs) < n_jobs and job_index < len(job_queue):
                job_index = submit_job(pool, job_queue, run_name, tariff_data, intermediate_filepaths, horizon_days, pending_jobs, job_index, infeasible_designs)
            
            # Process completed jobs and submit new ones
            while pending_jobs:
                # Wait for any job to complete
                for async_result in list(pending_jobs.keys()):
                    if async_result.ready():
                        job_idx, design_key = pending_jobs.pop(async_result)
                        result = async_result.get()
                        results.append(result)

                        if not result["is_feasible"]:
                            infeasible_designs.add(design_key)
                            print(f"Design {design_key} marked as {result['failure_reason']}, pruning smaller hours")
                        
                            # Remove unsubmitted jobs with this design key or smaller from the queue TODO: check
                            failed_hours = float(design_key.split('__')[0])
                            jobs_to_remove = []
                            for i, job in enumerate(job_queue):
                                job_design_key = job[0]
                                job_hours = float(job_design_key.split('__')[0])
                                if job_hours <= failed_hours and job_design_key not in infeasible_designs:
                                    jobs_to_remove.append(i)
                                    infeasible_designs.add(job_design_key)
                                    print(f"  Removing job for {job_design_key} from queue (≤{failed_hours} hours)")
                            
                            # Remove jobs in reverse order to maintain indices
                            for i in reversed(jobs_to_remove):
                                del job_queue[i]

                        # Submit next job if available
                        while job_index < len(job_queue):
                            job_index = submit_job(pool, job_queue, run_name, tariff_data, intermediate_filepaths, horizon_days, pending_jobs, job_index, infeasible_designs)
                            break

        # AFTER RUNNING all months / design keys, process NPV results for this iteration
        for Hours_of_O2, compression_ratio in unsolved_points:
            design_key = f"{Hours_of_O2}__{compression_ratio}"
            design_month_results = [r for r in results if r["design_key"] == design_key]

            if not design_month_results:  # create dummy results if all infeasible
                print(f"  No results processed for {design_key}, creating dummy infeasible result")
                design_month_results = [{
                    "design_key": design_key, "month_key": "2022-01", "month_results": None, 
                    "new_param_vals": None,  "is_feasible": False, "failure_reason": "Infeasible"
                }]

            annual_profiles = {"Edot_t_net": [], "Edot_t_baseline": [], "E": [], "N": [], "Ndot_c": []}
            max_values = {"Ndot_c": 0, "Edot_c": 0}
            all_dates = []
            for result in design_month_results:
                if not result["is_feasible"]:
                    continue
                    
                month_results = result["month_results"]
                
                # Get all days in this month to handle replacements for missing days
                if month_results:
                    first_date = next(iter(month_results.keys()))
                    year, month, _ = first_date.split("-")
                    all_days = get_all_days_in_month(int(year), int(month))
                    valid_days = [day for day, (profile, _) in month_results.items() if profile and "Edot_t_net" in profile]
                    for missing_date in set(all_days) - set(valid_days):
                        replacement_days = get_replacement_days(missing_date)
                        for replacement_day in replacement_days:
                            if replacement_day in valid_days:
                                print(f"Using {replacement_day} as replacement for {missing_date}")
                                replacement_profile, replacement_max_values = month_results[replacement_day]
                                for key in keys_to_append:
                                    if key in replacement_profile:
                                        annual_profiles[key].extend(replacement_profile[key])

                                for key in max_values:
                                    max_values[key] = max(max_values[key], replacement_max_values.get(key, 0))
                                break
                
                # Process actual solved days
                for profile, day_max_values in month_results.values():
                    if profile and "Edot_t_net" in profile:
                        for key in annual_profiles:
                            annual_profiles[key].extend(profile[key])
                        for key in max_values:
                            max_values[key] = max(max_values[key], day_max_values.get(key, 0))

                        for day in month_results.keys():
                            if month_results[day][0] == profile:  # This profile corresponds to this day
                                all_dates.append(day)
                                break

            num_months = sum(1 for result in design_month_results if result["is_feasible"])
            annual_metrics = calculate_metrics_from_annual_profiles(annual_profiles, tariff_key, upgrade_key, new_param_vals, tariff_data, num_months,  all_dates[0], all_dates[-1])
            annual_metrics["max_values"] = max_values

            # Calculate maximum values across all months for capex calculation
            sub_dict = {"param_vals": new_param_vals, "billing_key": tariff_key, "max_values": {}}
            for max_key in ["Ndot_c", "Edot_c"]:
                sub_dict["max_values"][max_key] = annual_metrics_data.get("max_values", {}).get(max_key, 0)
            
            capex, tank_metrics, capex_components, counterfactual_capex = (
                calculate_capex(
                    sub_dict=sub_dict,
                    storage_type=upgrade_key.split("__")[1],
                    new_o2_supply_tech=upgrade_key.split("__")[0],
                    base_wwtp_key=base_wwtp_key,
                    new_param_vals=new_param_vals,
                )
            )

            # Calculate NPV
            npv = calculate_itemized_npv(annual_metrics_data, capex)
            npv["from capex savings"] = counterfactual_capex
            print(
                f" {design_key} NPV energy ${npv['by_component']['energy']:,.2f} demand ${npv['by_component']['demand']:,.2f} export ${npv['by_component']['export']:,.2f} H2 ${npv['by_component']['h2']:,.2f} Total ${npv['total']:,.2f} capex {capex}"
            )

            # Store results
            all_results[design_key] = {
                "design_key": design_key,
                "metrics": annual_metrics_data,
                "month_profiles": None, # No longer storing monthly profiles
                "new_param_vals": new_param_vals,
            }

            # Clean up memory
            del annual_metrics_data
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
