import json
import os
import sys

# Limit linear algebra libraries to 1 thread per process for all parallel uses of IPOPT
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from helpers.tariffs import get_tariff_configs
from helpers.ingest_data import ingest_data
from helpers.plotting import generate_plots
from helpers.design_optimization_algorithm import run_configuration
from helpers.config_labels import get_unique_configurations, WWTP_TYPES
from helpers.parameters import get_summer_key

def extract_config_values(config):
    """Extract all configuration values from config dictionary."""

    if "design_point" in config:  # for sensitivity / tornado
        upgrade_keys = {config["base_wwtp_key"]: [config["upgrade_key"]]}
    else:
        upgrade_keys = config["upgrade_keys"]
    
    return {
        "base_run_name": config["run_name_from_file"],  # Use filename instead of config field
        "summer_config_template": config["summer_config_template"],
        "upgrade_keys": upgrade_keys,
        "tariff_keys_to_run": config["tariff_keys_to_run"],
        "shorten_months_run": config["data_ingestion"]["shorten_months_run"],
        "scale_factor": config["data_ingestion"]["scale_factor"],
        "ingest_gas": config["data_ingestion"]["ingest_gas"],
        "o2_range": config["design_space"]["o2_range"],
        "comp_ratio_range": config["design_space"]["comp_ratio_range"],
        "designs_per_run": config["design_space"]["designs_per_run"],
        "max_iterations": config["design_space"]["max_iterations"],
        "n_jobs": config["design_space"]["n_jobs"],
        "run_ingest_data": config["script_settings"]["run_ingest_data"],
        "run_solve_optimization": config["script_settings"]["run_solve_optimization"],
        "run_plotting": config["script_settings"]["run_plotting"],
        "figure_2": config["script_settings"]["figure_2"],
        "figure_4": config["script_settings"]["figure_4"],
        "figure_3": config["script_settings"]["figure_3"],
    }


def build_run_configs(summer_config_template, upgrade_keys, tariff_keys_to_run, config=None):
    """Build run configurations from templates."""
    run_configs = {}
    
    # Handle TORNADO configurations
    if config and "design_point" in config:
        base_wwtp_key = config["base_wwtp_key"]
        baseline_tariff = tariff_keys_to_run[0]  # assume first
        
        # Create separate run configs for each summer config
        for summer_config in summer_config_template:
            summer_config['suffix'] = get_summer_key(summer_config['multiplier'], summer_config['smoothing'])
            combined_base_wwtp_key = f"{base_wwtp_key}__{summer_config['multiplier']}__{summer_config['smoothing']}"
            run_configs[combined_base_wwtp_key] = {
                "upgrade_keys": [config["upgrade_key"]],
                "summer_configs": [summer_config],
                "tariff_configs": [baseline_tariff],
                "design_point": config["design_point"],
                "run_name": config["run_name_from_file"],
                "design_space": config["design_space"],
                "script_settings": config["script_settings"],
            }
        
        # Add one more config for all tariff variations with the baseline summer config
        baseline_summer = next((sc for sc in summer_config_template 
                               if sc['multiplier'] == 1.0 and sc['smoothing'] == 0), 
                              summer_config_template[0])
        
        combined_base_wwtp_key = f"{base_wwtp_key}__{baseline_summer['multiplier']}__{baseline_summer['smoothing']}_tariff_variations"
        run_configs[combined_base_wwtp_key] = {
            "upgrade_keys": [config["upgrade_key"]],
            "summer_configs": [baseline_summer],
            "tariff_configs": tariff_keys_to_run,
            "design_point": config["design_point"],
            "run_name": config["run_name_from_file"],
            "design_space": config["design_space"],
            "script_settings": config["script_settings"],
        }
        return run_configs
    
    # Handle WWTP COMPARISON configurations
    for wwtp_type in WWTP_TYPES:
        for config_item in summer_config_template:
            upgrades = upgrade_keys[wwtp_type]
            if upgrades:  # Only add if there are upgrades
                combined_base_wwtp_key = f"{wwtp_type}__{config_item['multiplier']}__{config_item['smoothing']}"
                run_configs[combined_base_wwtp_key] = {
                    "upgrade_keys": upgrades,
                    "summer_configs": [config_item],
                    "tariff_configs": tariff_keys_to_run,
                }
            else:
                print(f"Skipping {wwtp_type} because no upgrades defined")
    return run_configs


def main():
    # Load config
    config_file = sys.argv[1]  # command line argument for config
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, "r") as f:
        config = json.load(f)
    run_name_from_file = os.path.splitext(os.path.basename(config_file))[0]
    config["run_name_from_file"] = run_name_from_file
    config_values = extract_config_values(config)
        
    # Data ingestion
    if config_values["run_ingest_data"]:
        ingest_data(
            run_name=config_values["base_run_name"],
            ingest_gas=config_values["ingest_gas"],
            scale_factor=config_values["scale_factor"],
            shorten_months_run=config_values["shorten_months_run"],
        )

    # Optimization
    unique_configs = None
    if config_values["run_solve_optimization"]:
        run_configs = build_run_configs(
            config_values["summer_config_template"],
            config_values["upgrade_keys"],
            config_values["tariff_keys_to_run"],
            config
        )
        unique_configs = get_unique_configurations(run_configs)
        for config_item in unique_configs:
            run_configuration(
                config_item,
                config_values["base_run_name"],
                config_values["designs_per_run"],
                config_values["o2_range"],
                config_values["comp_ratio_range"],
                config_values["n_jobs"],
                config_values["max_iterations"],
                horizon_days=3
            )

    # Plotting
    if config_values["run_plotting"]:
        generate_plots(
            run_name=config_values["base_run_name"],
            run_configs=run_configs,
            location_lookup=get_tariff_configs(),
            figure_2=config_values["figure_2"],
            figure_4=config_values["figure_4"],
            figure_3=config_values["figure_3"],
            run_config=config
        )

if __name__ == "__main__":
    main()