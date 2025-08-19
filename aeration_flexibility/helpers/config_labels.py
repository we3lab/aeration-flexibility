from helpers.parameters import m3_hr_to_mgd
import re

plant_size = {"m3_hr": 4731, "MGD": 4731 * m3_hr_to_mgd}
WWTP_TYPES = ["air__compressor", "o2__psa", "o2__cryo"]

facility_labels = {
    "air__compressor": "Air from Blower",
    "o2__psa": "O$_2$ from PSA",
    "o2__cryo": "O$_2$ from Cryo",
}

upgrade_key_labels = {
    "elec__gas_tank": "Electrolyzer + Gas Tank",
    "none__gas_tank": "Liquid/Gas Tank for Excess O$_2$",
    "none__liquid_tank": "Liquid/Gas Tank for Excess O$_2$",
    "compressor__gas_tank": "Liquid/Gas Tank for Excess O$_2$",
    "none__battery": "Battery",
}

upgrade_key_colors = {
    "Electrolyzer + Gas Tank": 3,
    "Gas/Liquid O$_2$ Storage": 0,
    "Compressor + Gas Tank": 0,
    "Liquid/Gas Tank for Excess O$_2$": 0,
    "Battery": 5,
}

def parse_parametrized_tariff_name(tariff_key):
    """Parse parametrized tariff key to extract demand, energy, and window multipliers."""
    # Extract the base part after the prefix (e.g., "billing_d0.9_e1.0_wneg1")
    base_part = tariff_key.split("__", 1)[1] if "__" in tariff_key else tariff_key
    
    # Parse in order: d, e, w
    demand_match = re.search(r'd([0-9.]+)', base_part)
    energy_match = re.search(r'e([0-9.]+)', base_part)
    window_match = re.search(r'w(neg[0-9]+|[0-9]+)', base_part)

    if not demand_match:
        return {'demand_mult': 1.0, 'energy_mult': 1.0, 'window_change': 0}
    
    demand_mult = float(demand_match.group(1))
    energy_mult = float(energy_match.group(1)) if energy_match else 1.0  # TODO: check
    
    window_str = window_match.group(1) if window_match else '0'
    window_change = -int(window_str[3:]) if window_str.startswith('neg') else int(window_str)
    
    return {'demand_mult': demand_mult, 'energy_mult': energy_mult, 'window_change': window_change}

def get_tariff_key_labels(tariff_keys):
    """
    Generate readable labels for tariff keys based on the new naming convention.
    
    Args:
        tariff_keys: List of tariff keys
    
    Returns:
        dict: Mapping from tariff keys to readable labels
    """
    tariff_key_labels = {}
    
    for tariff_key in tariff_keys:
        # Baseline case
        if tariff_key.endswith("__billing") or tariff_key == "billing":
            tariff_key_labels[tariff_key] = "Baseline Tariff"
            continue
        
        # Parse parametrized tariff names
        parsed = parse_parametrized_tariff_name(tariff_key)
        
        # Build readable label
        parts = []
        if parsed['demand_mult'] != 1.0:
            parts.append(f"{int(parsed['demand_mult'] * 100)}% demand")
        if parsed['energy_mult'] != 1.0:
            parts.append(f"{int(parsed['energy_mult'] * 100)}% energy")
        if parsed['window_change'] != 0:
            sign = "+" if parsed['window_change'] > 0 else ""
            parts.append(f"{sign}{parsed['window_change']}h window")
        
        tariff_key_labels[tariff_key] = "Baseline Tariff" if not parts else ", ".join(parts)
    
    return tariff_key_labels


def get_unique_configurations(run_configs):
    unique_configs = []
    for base_wwtp_key in run_configs:
        parts = base_wwtp_key.split("__")
        base_wwtp_key_base = "__".join(parts[:2])
        run_config = run_configs[base_wwtp_key]
        
        for upgrade_key in run_config["upgrade_keys"]:
            tariff_configs_for_run = run_config["tariff_configs"]
            summer_configs_for_run = run_config["summer_configs"]
            if not isinstance(
                tariff_configs_for_run, list
            ):  # convert single tariff into list
                tariff_configs_for_run = [tariff_configs_for_run]
            # Ensure summer_configs_for_run is a list
            if not isinstance(summer_configs_for_run, list):
                summer_configs_for_run = [summer_configs_for_run]
            for tariff_key in tariff_configs_for_run:
                for summer_config in summer_configs_for_run:
                    summer_multiplier = float(summer_config["multiplier"])
                    summer_smoothing = int(summer_config["smoothing"])
                    
                    # Create base configuration with combined_base_wwtp_key
                    combined_base_wwtp_key = f"{base_wwtp_key_base}__{summer_multiplier}__{summer_smoothing}"
                    config = {
                            "base_wwtp_key": combined_base_wwtp_key,
                            "upgrade_key": upgrade_key,
                            "tariff_key": tariff_key,
                            "summer_config": summer_config,
                            "summer_multiplier": summer_multiplier,
                            "summer_smoothing": summer_smoothing,
                        }
                    
                    # Preserve any additional keys from the run configuration
                    # (needed for tornado analysis)
                    for key, value in run_config.items():
                        if key not in ["upgrade_keys", "tariff_configs", "summer_configs"]:
                            config[key] = value
                    
                    unique_configs.append(config)
    return unique_configs
