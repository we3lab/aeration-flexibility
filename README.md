# Aeration Flexibility

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

WE3Lab documentation for "Reducing the cost of wastewater aeration through optimally designed air and high-purity oxygen storage".

## About


### 1. Run with default configuration (standard analysis):
```bash
python run_all.py
```
This uses `configs/run_20250629.json` by default.

### 2. Run with a specific configuration:
```bash
python run_all.py configs/run_20250629.json
```

### 3. Run tornado analysis:
```bash
python run_all.py configs/run_test_nr_tornado.json
```

## File Structure

```
aeration_flexibility/
├── run_all.py                    # Main entry point for both analysis types
├── helpers/
│   ├── plotting.py              # Standard plotting functions (with tornado detection)
│   └── ...
├── configs/
│   ├── run_20250629.json        # Standard analysis configuration
│   └── run_test_nr_tornado.json # Tornado analysis configuration
└── output_data/                 # All output files
```


## Configuration File Structure

### Common Fields
- `run_name`: The name of the run
- `data_ingestion`: Data processing settings
- `script_settings`: Which parts of the pipeline to run

### WWTP Comparison Analysis Fields
- `summer_config_template`: Array of configurations with multiplier, smoothing, suffix
- `upgrade_keys`: Upgrade configurations for different WWTP types
- `design_space`: Ranges and parameters for multiple design points

### Tornado Analysis Fields
- `design_point`: Single design point [hours, compression_ratio]
- `base_wwtp_key`: WWTP type (e.g., "o2__psa")
- `upgrade_key`: Upgrade type (e.g., "none__gas_tank__0")
- `parametrized_variations`: Dictionary mapping variation names to tariff keys

## Example Configurations

### Standard Analysis (`configs/run_20250629.json`)
```json
{
  "run_name": "run_20250629",
  "summer_config_template": [
    {
      "multiplier": 1.2,
      "smoothing": 2,
      "suffix": "nr_smooth"
    },
    {
      "multiplier": 1.0,
      "smoothing": 0
    }
  ],
  "upgrade_keys": {
    "air__compressor": [],
    "o2__psa": ["elec__gas_tank__0", "none__gas_tank__0"],
    "o2__cryo": ["elec__gas_tank__0", "none__liquid_tank__0"]
  },
  "data_ingestion": {
    "shorten_months_run": 1,
    "scale_factor": 2.963
  },
  "design_space": {
    "o2_range": [0.25, 3.0],
    "comp_ratio_range": [10.0, 700.0],
    "designs_per_run": 5,
    "max_iterations": 5,
    "n_jobs": 20
  },
  "script_settings": {
    "run_ingest_data": true,
    "run_solve_optimization": true,
    "run_plotting": true
  }
}
```

### Tornado Analysis (`configs/run_test_nr_tornado.json`)
```json
{
  "run_name": "run_test_nr_tornado",
  "design_point": [1.0, 100.0],
  "base_wwtp_key": "o2__psa",
  "upgrade_key": "none__gas_tank__0",
  "parametrized_variations": {
    "baseline": "0.0__billing",
    "decrease_peak": "0.0__decreased_demand",
    "increase_peak": "0.0__increased_demand",
    "decrease_energy": "0.0__decreased_energy",
    "increase_energy": "0.0__increased_energy",
    "decrease_window": "0.0__decreased_peak_window",
    "increase_window": "0.0__increased_peak_window"
  },
  "data_ingestion": {
    "shorten_months_run": 1,
    "scale_factor": 2.963
  },
  "design_space": {
    "o2_range": [0.25, 3.0],
    "comp_ratio_range": [10.0, 700.0],
    "designs_per_run": 1,
    "max_iterations": 1,
    "n_jobs": 7
  },
  "script_settings": {
    "run_ingest_data": false,
    "run_solve_optimization": true,
    "run_plotting": true
  }
}
```

## Data Ingestion

The package utilized functions from the WE3 Lab's proprietary flows-prep package for processing:
- prep_raw_data() → Detotalizes raw SCADA data and calculates virtual tags
- prep_clean_data() → Cleans the data
- prep_imputed_data() → Imputes missing values

Virtual tags are calculated during this stage
(e.g., VirtualDemand_Electricity_InFlow, VirtualDemand_RestOfFacilityPower)

Scaling happens after virtual tags are calculated
The scaling is applied to the final cleaned data that already contains the virtual tags