import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
import pickle
import os
import glob

from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib import ticker
from scipy.interpolate import griddata, RBFInterpolator
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from helpers.design_optimization_algorithm import calculate_itemized_npv
from helpers.parameters import *
from helpers.config_labels import *


def npv_to_delta_lcot(npv, years=20, mgd=PLANT_CAPACITY_MGD, baseline_lcot=1.00):
    """Convert NPV to delta LCOT as percentage change relative to baseline LCOT"""
    m3_per_day = mgd * MGD_TO_M3_PER_DAY
    total_m3 = m3_per_day * 365 * years
    delta_lcot_dollars = -npv / total_m3
    return (delta_lcot_dollars / baseline_lcot) * 100  # Convert to percentage


def is_valid_npv(npv):
    """Check if NPV is a valid finite number."""
    if npv is None:
        print(f" NPV is None")
        return False
    if isinstance(npv, (np.ndarray, list)):
        if len(npv) == 0:
            print(f" NPV is empty array/list")
            return False
        is_valid = np.isfinite(npv).all()
        return is_valid
    is_valid = np.isfinite(float(npv))
    return is_valid


def get_output_dir(run_name, subdir=None):
    """Get standardized output directory path."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "output_plots", run_name)
    if subdir:
        output_dir = os.path.join(output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_results_from_files(run_name):
    """Load and organize results from pickle files."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_data")
    pattern = f"{data_dir}/{run_name}/*.pkl"
    found_files = glob.glob(pattern)
    results = {}
    for file_path in found_files:
        if "___" in file_path:
            filename = os.path.basename(file_path)
            config_key = filename.replace(".pkl", "")
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                results[config_key] = data
    return results


def get_hatch_patterns():
    """Get standardized hatch patterns for different components."""
    return {"CAPEX": "", "Energy": "||||", "Demand": "////", "Export": "++++", "H$_2$": "XXXX"}


def get_upgrade_label_mappings():
    """Get mappings between upgrade keys and their display labels for different contexts."""
    mappings = {
        # For heatmap titles (subplots D, E, F) - use nice display labels
        "elec__gas_tank": "Electrolyzer + Gas Tank",
        "compressor__gas_tank": "Compressor + Gas Tank", 
        "none__gas_tank": "Gas Tank for Excess O₂",
        "none__liquid_tank": "Liquid Tank for Excess O₂",
        "none__battery": "Battery"
    }
    return mappings


def get_design_data(config_data):
    """Extract all_results and best design data from config_data."""    
    all_results = config_data["all_results"]
    valid_keys = get_valid_design_keys(all_results)
    
    if not valid_keys:
        # TODO: remove if solves are working
        print('No valid keys found')
        dummy_key = list(all_results.keys())[0] if all_results else None
        return all_results, dummy_key
    
    best_design_key = max(
        valid_keys, key=lambda k: all_results[k]["npv"]["total"]
    )
    return all_results, best_design_key


def get_custom_lcot_cmap(n_levels=60, is_npv=False):
    """
    Args:
        n_levels (int): Number of color levels in the colormap (increased for smoother transitions)
        is_npv (bool): If True, swap colors for NPV plots (red=negative, green=positive)
    """
    if is_npv:
        colors = [cb_palette[8], 'white', cb_palette[2]]
        positions = [0, 0.5, 1]
    else:
        colors = [cb_palette[2], 'white', cb_palette[8]]
        positions = [0, 0.5, 1]
    cmap = LinearSegmentedColormap.from_list("custom_lcot", list(zip(positions, colors)), N=n_levels)
    cmap.set_bad("black")
    return cmap


def _plot_lcot_grid(hours, ratios, lcots, grid_size=400):
    """Create a grid of LCOT values with smooth interpolation."""

    hours = np.array(hours)
    ratios = np.array(ratios)
    lcots = np.array(lcots)
    mask = np.isfinite(lcots)
    hours, ratios, lcots = hours[mask], ratios[mask], lcots[mask]

    # Check if we have any data to work with
    if len(hours) == 0 or len(ratios) == 0 or len(lcots) == 0:
        print("Warning: Empty data arrays provided to _plot_lcot_grid")
        return None, None, None
    # Check if all values are NaN and set them to 0
    if np.all(np.isnan(lcots)):
        print("Warning: All LCOT values are NaN, setting to 0")
        lcots = np.zeros_like(lcots)
    # If all values are NaN, set them to 0
    if not np.any(mask):
        print("Warning: All LCOT values are NaN, setting to 0")
        lcots = np.zeros_like(lcots)
        mask = np.ones_like(lcots, dtype=bool)
    # Check if we have any valid data after masking
    if len(hours) == 0 or len(ratios) == 0 or len(lcots) == 0:
        print("Warning: No valid finite values found after masking")
        return None, None, None
    # Check if we have enough points for interpolation (need at least 3)
    if len(hours) < 3:
        print(f"Warning: Insufficient data points ({len(hours)}) for interpolation, need at least 3")
        return None, None, None

    xi = np.linspace(np.min(hours), np.max(hours), grid_size)
    yi = np.linspace(np.min(ratios), np.max(ratios), grid_size)
    xi, yi = np.meshgrid(xi, yi)
    
    # Try RBF interpolation
    try:
        # Normalize coordinates for better RBF performance
        hours_range = np.max(hours) - np.min(hours)
        ratios_range = np.max(ratios) - np.min(ratios)
        hours_norm = (hours - np.min(hours)) / hours_range
        ratios_norm = (ratios - np.min(ratios)) / ratios_range
        xi_norm = (xi - np.min(hours)) / hours_range
        yi_norm = (yi - np.min(ratios)) / ratios_range
        
        # Stack normalized coordinates
        points = np.column_stack([hours_norm, ratios_norm])
        grid_points = np.column_stack([xi_norm.flatten(), yi_norm.flatten()])
        
        # Use RBF interpolation with thin plate spline
        rbf = RBFInterpolator(points, lcots, kernel='thin_plate_spline')
        zi = rbf(grid_points).reshape(xi.shape)
        
    except Exception:
        # griddata with cubic interpolation
        try:
            zi = griddata((hours, ratios), lcots, (xi, yi), method='cubic', fill_value=np.nanmean(lcots))
        except Exception:
            # Final fallback to linear interpolation
            zi = griddata((hours, ratios), lcots, (xi, yi), method='linear', fill_value=np.nanmean(lcots))
    
    return xi, yi, zi


def setup_plot_style(ax, title=None, xlabel=None, ylabel=None, fontsize=12):
    if title:
        ax.set_title(title, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(True, alpha=0.3)


def plot_stacked_bar(ax, pos, values, width, color, hatch, **kwargs):
    """Plot a stacked bar with common styling."""
    return ax.bar(
        pos, values, width, color=color, hatch=hatch,
        edgecolor="black", linewidth=1.5, alpha=0.8, **kwargs
    )


def parse_config_key(config_key):
    """Parse config key into base_wwtp_key, upgrade_key, tariff_key."""
    # Split by ___ to get the parts
    parts = config_key.split("___")
    
    # If we have 4 parts, the last one is the suffix
    return parts[0], parts[1], parts[2], parts[3]


def get_valid_design_keys(all_results):
    """Get list of design keys with valid NPV."""
    valid_keys = []
    
    for k, result in all_results.items():
        npv = result.get("metrics", {}).get("npv", {}).get("total")
        if is_valid_npv(npv):
            valid_keys.append(k)
        else:
            print(f" Invalid NPV for {k}: {npv}")
    return valid_keys


def _get_color_config():
    """Get configuration for energy plot styling."""
    return {
        "facility_markers": {
            "air__compressor": "o",
            "o2__psa": "s", 
            "o2__cryo": "^",
        },
        "upgrade_colors": {
            "none__battery": cb_palette[upgrade_key_colors["Battery"]],
            "none__gas_tank": cb_palette[upgrade_key_colors["Gas/Liquid O$_2$ Storage"]],
            "none__liquid_tank": cb_palette[upgrade_key_colors["Gas/Liquid O$_2$ Storage"]],
            "elec__gas_tank": cb_palette[upgrade_key_colors["Electrolyzer + Gas Tank"]],
            "compressor__gas_tank": cb_palette[upgrade_key_colors["Compressor + Gas Tank"]],
        }
    }


def _extract_energy_data(results, tariff_key):
    """Extract energy metrics data from results."""
    energy_data = []
    
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, config_tariff_key, suffix = parse_config_key(config_key)
        if config_tariff_key != tariff_key:
            continue

        all_results = (config_data["all_results"] if isinstance(config_data, dict) and "all_results" in config_data 
                      else config_data)
        
        valid_keys = get_valid_design_keys(all_results)
        if not valid_keys:
            print(f"No valid keys for {config_key}, skipping.")
            continue

        best_design_key = max(valid_keys, key=lambda k: all_results[k]["npv"]["total"])
        design_data = all_results[best_design_key]

        if "mean_energy_capacity" not in design_data:
            print("mean_energy_capacity not found in data")
            continue

        capacity_factor = design_data["capacity_factor"]
        mean_energy_capacity = design_data["mean_energy_capacity"]
        mean_power_capacity = design_data["mean_power_capacity"]

        if np.isnan([capacity_factor, mean_energy_capacity, mean_power_capacity]).any():
            print(f"Skipping {config_key} due to NaN in energy metrics.")
            continue

        facility_type = base_wwtp_key.split("__")[0] + "__" + base_wwtp_key.split("__")[1]
        base_upgrade_key = "__".join(upgrade_key.split("__")[:2])
        
        energy_data.append({
            "capacity_factor": capacity_factor,
            "mean_energy_capacity": mean_energy_capacity,
            "mean_power_capacity": mean_power_capacity,
            "facility_type": facility_type,
            "base_upgrade_key": base_upgrade_key
        })
    
    return energy_data


def _plot_energy_metrics(ax_energy, ax_energy2, energy_data, color_config):
    """Plot energy metrics on the given axes."""
    facility_markers = color_config["facility_markers"]
    upgrade_colors = color_config["upgrade_colors"]
    plotted_points = []

    for data in energy_data:
        marker = facility_markers.get(data["facility_type"], "o")
        color = upgrade_colors.get(data["base_upgrade_key"], "#000000")

        # Plot energy capacity (solid marker)
        ax_energy.scatter(data["capacity_factor"], data["mean_energy_capacity"], marker=marker,
                         color=color, s=200, alpha=0.9, label=f"{data['facility_type']} Energy")
        
        # Plot power capacity (hollow marker)
        ax_energy2.scatter(data["capacity_factor"], data["mean_power_capacity"], marker=marker,
                          color=color, s=200, facecolors="none", alpha=0.9, linewidth=2,
                          label=f"{data['facility_type']} Power")
        
        plotted_points.append((marker, color, data["facility_type"]))

    return plotted_points


def plot_combined_metrics(results, run_name, output_dir, tariff_key="0.0__svcw", location_lookup=None):
    """Plot combined metrics showing LCOT bars and energy metrics in a 1x2 subplot layout."""
    # Create figure
    fig, (ax_energy, ax_lcot) = plt.subplots(1, 2, figsize=(20, 8))
    ax_energy2 = ax_energy.twinx()

    # Setup energy plot styling
    setup_plot_style(ax_energy, xlabel="Capacity Factor\n(max % of storage)", 
                     ylabel="Normalized\nEnergy Capacity", fontsize=24)
    setup_plot_style(ax_energy2, ylabel="Normalized\nPower Capacity", fontsize=20)
    ax_energy.set_ylim(0.0, 1.0)
    ax_energy2.set_ylim(0.0, 1.0)
    ax_energy.grid(True, linestyle="--", alpha=0.7)  

    # Extract and plot energy data
    energy_data = _extract_energy_data(results, tariff_key)
    color_config = _get_color_config()
    plotted_points = _plot_energy_metrics(ax_energy, ax_energy2, energy_data, color_config)

    # Plot LCOT bars
    npv_data = consolidate_npv_data(results, {})
    plot_npv_bars(npv_data, ax=ax_lcot)

    # Save figure
    plt.subplots_adjust(wspace=0.4)  # Increased horizontal gap between subplots
    save_path = os.path.join(output_dir, "combined_metrics.png")
    plt.savefig(save_path, bbox_inches="tight")


def consolidate_npv_data(results, grid_results):
    """Consolidate NPV data from all configurations"""
    npv_data = []
    for config_key, config_data in results.items():     
        all_results, best_design_key = get_design_data(config_data)

        if all_results is None or best_design_key is None:
            print(f"Skipping {config_key}: No valid design data found")
            continue

        design_data = all_results[best_design_key]        
        if design_data is None:
            print(f"Skipping {config_key}: design_data is None")
            continue
            
        npv_metrics = design_data["npv"]
        
        if "by_component" not in npv_metrics: #TODO: remove
            print(f"Skipping {config_key}: NPV structure incomplete (missing 'by_component')")
            npv_components =  {
                                    "energy": -np.inf,
                                    "demand": -np.inf,
                                    "export": -np.inf,
                                    "h2": -np.inf
                                }
        else:
            npv_components = npv_metrics["by_component"]
        
        # Extract NPV components
        components = {
            "energy_npv": npv_components.get("energy"),
            "demand_npv": npv_components.get("demand"), 
            "export_npv": npv_components.get("export"),
            "h2_npv": npv_components.get("h2"),
            "capex_npv": -design_data.get("capex"),
            "npv": npv_metrics.get("total")
        }
        
        # Parse config key to get facility and upgrade info
        base_wwtp_key, upgrade_key, tariff_key, suffix = parse_config_key(config_key)
        facility = facility_labels[base_wwtp_key.split("__")[0] + "__" + base_wwtp_key.split("__")[1]]
        upgrade = upgrade_key_labels[upgrade_key.split("__")[0] + "__" + upgrade_key.split("__")[1]]
        
        # Validate component NPVs
        if not all(is_valid_npv(x) for x in components.values()):
            print(f"Skipping {config_key} due to invalid NPV components: {list(components.values())}")
            continue
        
        # Parse design key for hours and compression ratio
        hours, comp_ratio = map(float, best_design_key.split("__"))
        
        npv_data.append({
            "Facility": facility,
            "Upgrade Type": upgrade,
            "Rate Config": tariff_key,
            **components,
            "Hours of O2": hours,
            "Compression Ratio": comp_ratio,
        })

    if not npv_data:
        print("No valid NPV data found for any configuration")
        return pd.DataFrame()

    return pd.DataFrame(npv_data)


def plot_npv_bars(npv_data, ax=None):
    """Plot NPV bars for a given configuration."""
    fig = ax.figure
    plot_df = npv_data.copy()
    # Convert NPV components to LCOT
    lcot_columns = ["energy_npv", "demand_npv", "export_npv", "h2_npv", "capex_npv", "npv"]
    # lcot_columns = ["energy_npv", "demand_npv", "h2_npv", "capex_npv", "npv"]

    for col in lcot_columns:
        plot_df[f"Delta LCOT from {col.replace('_npv', '').title()}"] = plot_df[col].apply(npv_to_delta_lcot)

    # Recalculate total from components
    plot_df["Delta LCOT Total"] = sum(
        plot_df[f"Delta LCOT from {col.replace('_npv', '').title()}"] for col in lcot_columns[:-1]
    )

    x_categories = plot_df['Facility'].unique()
    desired_order = ["Air from Blower", "O$_2$ from PSA", "O$_2$ from Cryo"]
    x_categories = [cat for cat in desired_order if cat in x_categories]
    n_x = len(x_categories)
    fig_width = max(8, 1.5 * n_x)
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, 8))
    width = min(0.8, 0.8 * 8 / n_x)
    upgrade_types = plot_df["Upgrade Type"].unique()
    x_positions = np.arange(len(x_categories))
    x_labels = list(x_categories)
    ax.axhline(0, color="black", lw=0.5)
    highest_point, lowest_point, h2_nonzero = 0, 0, False
    upgrade_handles = []

    for i, upgrade in enumerate(upgrade_types):
        upgrade_data = plot_df[plot_df["Upgrade Type"] == upgrade]
        color = cb_palette[upgrade_key_colors[upgrade]]
        pos = []
        comp_arrays = {k: [] for k in ["CAPEX", "Energy", "Demand", "Export", "H$_2$"]}
        total_lcot_data, hours_data = [], []
        for j, x_cat in enumerate(x_categories):
            x_cat_data = upgrade_data[upgrade_data["Facility"] == x_cat]
            if not x_cat_data.empty:
                pos.append(x_positions[j] + (i - len(upgrade_types) / 2 + 0.5) * width / len(upgrade_types))
                comp_arrays["CAPEX"].append(x_cat_data["Delta LCOT from Capex"].iloc[0])
                comp_arrays["Energy"].append(x_cat_data["Delta LCOT from Energy"].iloc[0])
                comp_arrays["Demand"].append(x_cat_data["Delta LCOT from Demand"].iloc[0])
                comp_arrays["Export"].append(x_cat_data["Delta LCOT from Export"].iloc[0])
                comp_arrays["H$_2$"].append(x_cat_data["Delta LCOT from H2"].iloc[0])
                total_lcot_data.append(x_cat_data["Delta LCOT Total"].iloc[0])
                hours_data.append(x_cat_data["Hours of O2"].iloc[0])
                if x_cat_data["Delta LCOT from H2"].iloc[0] != 0.0:
                    h2_nonzero = True
        for idx in range(len(pos)):
            pos_bottom, neg_bottom = 0, 0
            for label in ["CAPEX", "Energy", "Demand", "Export", "H$_2$"]:
                value = comp_arrays[label][idx]
                # Only apply H2 hatching if there's actually an H2 component
                if label == "H$_2$" and not h2_nonzero:
                    hatch = ""  # No hatching for H2 when there's no H2 component
                else:
                    hatch = get_hatch_patterns()[label]
                if value < 0:
                    plot_stacked_bar(ax, pos[idx], value, width / len(upgrade_types), color, hatch, bottom=neg_bottom)
                    neg_bottom += value

            # Then plot positive components
            for label in ["CAPEX", "Energy", "Demand", "Export", "H$_2$"]:
                value = comp_arrays[label][idx]
                # Only apply H2 hatching if there's actually an H2 component
                if label == "H$_2$" and not h2_nonzero:
                    hatch = ""  # No hatching for H2 when there's no H2 component
                else:
                    hatch = get_hatch_patterns()[label]
                if value >= 0:
                    plot_stacked_bar(ax, pos[idx], value, width / len(upgrade_types), color, hatch, bottom=pos_bottom)
                    pos_bottom += value
            highest_point = max(highest_point, pos_bottom)
            lowest_point = min(lowest_point, neg_bottom)

        # Plot dots using original total
        ax.scatter(pos, total_lcot_data, color="black", s=75, zorder=3)

        # Plot hours text
        for idx, (p, total, hours) in enumerate(zip(pos, total_lcot_data, hours_data)):
            components_sum = max(0, sum(comp_arrays[label][idx] for label in ["CAPEX", "Energy", "Demand", "Export", "H$_2$"] if comp_arrays[label][idx] > 0))
            text_position = components_sum + 0.0005
            ax.text(p, text_position, f"{round(hours,2)}", ha="center", va="bottom", fontsize=20)
            highest_point = max(highest_point, text_position + 0.0005)
        upgrade_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, label=upgrade, edgecolor="black", linewidth=1.5)
        )
    setup_plot_style(ax, ylabel="Change in LCOT (%)", fontsize=18)
    # Position y-label consistently with subplots A and D
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_ylim(lowest_point - 0.01, highest_point * 1.25)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=20)
    cost_handles_filtered = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=1.5, hatch=get_hatch_patterns()[label], label=label)
        for label in ["CAPEX", "Energy", "Demand", "Export", "H$_2$"] if not (label == "H$_2$" and not h2_nonzero)
    ]
    cost_handles_filtered.append(
        plt.Line2D([0], [0], marker="o", color="w",
        markerfacecolor="black", markersize=12,
        markeredgecolor="black", markeredgewidth=1.5,
        label="Total Delta LCOT (%)")
    )
    all_handles = upgrade_handles + cost_handles_filtered
    upgrade_cols = len(upgrade_handles)

    def custom_patch_func(
        legend, orig_handle, xdescent, ydescent, width, height, fontsize
    ):
        """Custom function to draw patches with thicker hatch patterns.
        Created by Claude"""
        p = plt.Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
            hatch=orig_handle.get_hatch(),
        )
        hatch = orig_handle.get_hatch()
        if hatch is not None:
            p.set_hatch(hatch * 2)  # Double the hatch density
        return p

    ax.legend(
        handles=all_handles,
        bbox_to_anchor=(0.5, 1.08),
        loc="upper center",
        ncol=max(upgrade_cols, 3),
        frameon=False,
        fontsize=24,
        bbox_transform=plt.gcf().transFigure,
        handler_map={plt.Rectangle: HandlerPatch(patch_func=custom_patch_func)},
    )


def si_cem_figure(results, run_name, output_dir):
    """Plot sampling history for all configurations."""
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, tariff_key, suffix = parse_config_key(config_key)
        if "gas_tank" not in upgrade_key:
            continue

        sampled_points_history = config_data["sampled_points_history"]
        mu_sigma_history = config_data["mu_sigma_history"]

        all_results, best_design_key = get_design_data(config_data)
        if best_design_key is None:
            continue
        # Gather all valid points
        def valid_point(key):
            return (
                key in all_results and
                "metrics" in all_results[key] and
                "npv" in all_results[key] and
                is_valid_npv(all_results[key]["npv"]["total"])
            )
        def get_lcot(key):
            npv = all_results[key]["npv"]["total"]
            return npv_to_delta_lcot(npv)
        # Helper for plotting a single iteration
        def plot_iter(ax, hours, ratios, lcots, cmap, absmax, smoothing):
            xi, yi, zi = _plot_lcot_grid(hours, ratios, lcots)
            if xi is None or yi is None or zi is None:
                return False
            c = ax.pcolormesh(xi, yi, zi, shading="auto", cmap=cmap, vmin=-absmax, vmax=absmax)
            ax.scatter(hours, ratios, c=lcots, cmap=cmap, vmin=-absmax, vmax=absmax, edgecolor="black", s=50)
            setup_plot_style(ax, xlabel="Hours of O$_2$ Storage", ylabel="Compression Ratio")
            return c
        # Get all valid points for the config
        all_keys = [k for k in all_results if valid_point(k)]
        all_hours = [float(k.split("__")[0]) for k in all_keys]
        all_ratios = [float(k.split("__")[1]) for k in all_keys]
        all_lcots = [get_lcot(k) for k in all_keys]
        absmax = np.nanmax(np.abs(all_lcots))
        cmap = get_custom_lcot_cmap()
        # n_iter = len(sampled_points_history)
        for smoothing, label in zip([True, False], ["smoothed", "non_smoothed"]):
            valid_iters = []
            for i, (mu_sigma, sampled_points) in enumerate(zip(mu_sigma_history, sampled_points_history)):
                pts = [pt for sublist in sampled_points_history[:i+1] for pt in sublist]
                iter_keys = [f"{pt[0]}__{pt[1]}" for pt in pts if f"{pt[0]}__{pt[1]}" in all_results and valid_point(f"{pt[0]}__{pt[1]}")]
                if len(iter_keys) >= 3:
                    iter_hours = [float(k.split("__")[0]) for k in iter_keys]
                    iter_ratios = [float(k.split("__")[1]) for k in iter_keys]
                    iter_lcots = [get_lcot(k) for k in iter_keys]
                    valid_iters.append((i, iter_hours, iter_ratios, iter_lcots))
            if valid_iters:
                fig, axes = plt.subplots(1, len(valid_iters), figsize=(6 * len(valid_iters) + 2, 6), sharex=True, sharey=True)
                if len(valid_iters) == 1:
                    axes = [axes]
                for idx, (i, hours, ratios, lcots) in enumerate(valid_iters):
                    ax = axes[idx]
                    c = plot_iter(ax, hours, ratios, lcots, cmap, absmax, smoothing)
                    ax.set_title(f"Iteration {i+1}")
                    if idx == len(valid_iters) - 1 and c is not False:
                        cbar = fig.colorbar(c, ax=ax, label="Delta LCOT ($/m³)", pad=0.02)
                        cbar.ax.tick_params(labelsize=14)
                        cbar.ax.set_ylabel("Delta LCOT ($/m³)", fontsize=18)

                plt.tight_layout()
                plt.savefig(os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "output_plots", run_name, "mvn", 
                    f"mvn_exploration_{base_wwtp_key}__{upgrade_key}__{tariff_key}_{label}.png"), bbox_inches="tight")
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                c = plot_iter(ax, all_hours, all_ratios, all_lcots, cmap, absmax, smoothing)
                wwtp_type = (
                    "Conventional"
                    if "air" in base_wwtp_key
                    else f'HPOAS with O2 from {"PSA" if "psa" in base_wwtp_key else "Cryo"}'
                )
                ax.set_title(f'{wwtp_type} Facility with {upgrade_key_labels[upgrade_key.split("__")[0]+"__"+upgrade_key.split("__")[1]]}')
                if c is not False:
                    fig.colorbar(c, ax=ax, orientation="vertical", label="Delta LCOT ($/m³)")
                plt.tight_layout()
                plt.savefig(os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "output_plots", run_name, "mvn", 
                    f"mvn_exploration_{base_wwtp_key}__{upgrade_key}__{tariff_key}_{label}.png"))


def _add_storage_tank(ax, x, y, side, volume, label, color, zorder_offset=0):
    rect = plt.Rectangle((x, y), side, side, facecolor=color, zorder=2+zorder_offset)
    ax.add_patch(rect)
    ax.text(x + side/2, y + side*2, f"{label}\n{volume:,.0f} m³",
            ha="center", va="center", color="black", fontsize=16, 
            weight="bold", zorder=6+zorder_offset)


def plot_storage_vs_bioreactor_volume(results, run_name, output_dir, tariff_key="0.0__svcw"):
    """Plot storage volumes compared to bioreactor volume."""
    V_bioreactor = Q * HRT_DAYS
    grouped = {}
    
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, this_tariff_key, suffix = config_key.split("___")
        if this_tariff_key != tariff_key:
            continue
        if "gas_tank" not in upgrade_key and "liquid_tank" not in upgrade_key:
            continue
        all_results, best_design_key = get_design_data(config_data)
        design_data = all_results["1.0__100.0"]
        param_values = design_data["new_param_vals"]
        V_o2 = param_values["V_tank"]

        # Calculate H$_2$ volume if electrolyzer is present
        V_h2 = None
        if "elec" in upgrade_key:
            N_max_h2 = param_values.get("N_max", 0) * 2  # H$_2$ moles
            P_max_h2 = 35e6  # 35 MPa in Pa
            V_h2 = (N_max_h2 * 8.314 * 293) / P_max_h2  # Ideal gas law, T=20C
        grouped.setdefault(base_wwtp_key, {}).setdefault(upgrade_key, []).append((V_o2, V_h2))

    for base_wwtp_key, upgrade_dict in grouped.items():
        for upgrade_key, storage_data in upgrade_dict.items():
            if not storage_data:
                print(f"No valid storage volumes to plot for {base_wwtp_key} with {upgrade_key}.")
                continue
            
            V_o2, V_h2 = storage_data[0]
            max_vol = max([V_bioreactor] + [V_o2 if V_o2 is not None else 0] + [V_h2 if V_h2 is not None else 0])
            scale = 1.0 / np.sqrt(max_vol)
            min_side = 0.02
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Bioreactor
            side_bio = np.sqrt(V_bioreactor) * scale
            rect_bio = plt.Rectangle((0, 0), side_bio, side_bio, facecolor="#e0e0e0", zorder=0)
            ax.add_patch(rect_bio)
            ax.text(side_bio/2, side_bio/2, f"Bioreactor\n{V_bioreactor:,.0f} m³",
                    ha="center", va="center", fontsize=20, weight="bold", color="black", zorder=5)

            # O2 tank
            if V_o2 is not None:
                side_o2 = max(np.sqrt(V_o2) * scale, min_side)
                _add_storage_tank(ax, 0.02, 0.02, side_o2, V_o2, "O₂ Storage", "#1f77b4", 0)
            
            # H$_2$ tank
            if V_h2 is not None and V_h2 > 0:
                side_h2 = max(np.sqrt(V_h2) * scale, min_side)
                h2_x = side_bio - side_h2 - 0.02
                _add_storage_tank(ax, h2_x, 0.02, side_h2, V_h2, "H₂ Storage", "orange", 1)

            ax.set_aspect("equal")
            ax.axis("off")
            upgrade_key_base = "__".join(upgrade_key.split("__")[:2])
            setup_plot_style(ax, title=f"\n{upgrade_key_labels[upgrade_key_base]}", fontsize=20)
            plt.savefig(os.path.join(output_dir, f"storage_vs_bioreactor_area_{base_wwtp_key}_{upgrade_key}.png"))


def _figure_2_a_c(ax, day_data, data_config, with_recovery, upgrade_key, ylim=None, yticks=None):
    """Plot power profile with only the difference between baseline and optimized power filled."""
    time = [i / timestep_factor for i in range(1, int(24 * timestep_factor) + 1)]
    
    # Plot baseline and optimized lines first
    sns.lineplot(x=time, y=day_data["Edot_t_baseline"]/1000, color="grey", 
                linewidth=2, linestyle="--", ax=ax, label="Baseline")
    if with_recovery:
        sns.lineplot(x=time, y=day_data["Edot_t_net"]/1000, color="k", 
                    linewidth=2, linestyle="-", ax=ax, label="Optimized")
    
    # Calculate the difference between optimized and baseline
    baseline_power = day_data["Edot_t_baseline"]/1000
    optimized_power = day_data["Edot_t_net"]/1000 if with_recovery else baseline_power
    
    # Find where optimized power is different from baseline
    power_diff = optimized_power - baseline_power
    
    # Get upgrade type for color mapping
    upgrade_type = upgrade_key.split("__")[0] + "__" + upgrade_key.split("__")[1]
    upgrade_label = upgrade_key_labels[upgrade_type]
    positive_mask = power_diff > 0
    negative_mask = power_diff < 0
    
    ax.fill_between(time, baseline_power, optimized_power, 
                    where=positive_mask, color=cb_palette[8], alpha=0.8, 
                    label="Power Increase")
    
    ax.fill_between(time, optimized_power, baseline_power, 
                where=negative_mask, color=cb_palette[2], alpha=0.8, 
                label="Power Decrease")
    
    setup_plot_style(ax, ylabel="Power (MW)", fontsize=20)
    ax.set_xticks([])
    ax.set_xlim([0, 24])
    
    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.legend().set_visible(False)


def _figure_2_capex_scatter(hours_list, comp_ratio_list, capex_list, base_name, output_dir):
    """
    Helper function to create scatter plot of capex vs hours of storage.
    """
    valid_indices = [i for i, (h, c, cap) in enumerate(zip(hours_list, comp_ratio_list, capex_list)) 
                    if h > 0 and c > 0 and np.isfinite(cap)]
        
    hours_valid = [hours_list[i] for i in valid_indices]
    comp_ratio_valid = [comp_ratio_list[i] for i in valid_indices]
    capex_valid = [capex_list[i] for i in valid_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color points by compression ratio
    scatter = ax.scatter(hours_valid, capex_valid, c=comp_ratio_valid, 
                        cmap='viridis', alpha=0.7, s=50)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Compression Ratio", fontsize=14)
    
    setup_plot_style(ax, 
                     xlabel="Hours of Storage", 
                     ylabel="Capital Expenditure (US$M)", 
                     fontsize=14)
    out_path = os.path.join(output_dir, f"capex_vs_storage_scatter_{base_name}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()  # Close the figure to prevent interference with other plots


def _figure_2_power_config(upgrade_key, base_wwtp_key, with_storage, with_recovery):
    """Get power data configuration with upgrade-specific colors for storage components."""
    gas = base_wwtp_key.split("__")[0]
    blower_label = "From Blower" if gas == "air" else "From HPO Plant"
    
    upgrade_type = upgrade_key.split("__")[0] + "__" + upgrade_key.split("__")[1]
    
    if "battery" in upgrade_key:
        return [
            ("Edot_rem", 1, cb_palette[6], "Rest of Facility"),
            ("Edot_b", 1, cb_palette[0], blower_label),
            ("Edot_c", 1, cb_palette[2], "To Storage"),  # Charging: cb_palette[1]
            ("Edot_r", -1, cb_palette[8], "From Storage"),  # Discharging: upgrade color
        ]
    
    if with_recovery and with_storage:
        return [
            ("Edot_rem", 1, cb_palette[6], "Rest of Facility"),
            ("Edot_b", 1, cb_palette[0], blower_label),
            ("Edot_c", 1, cb_palette[2], "To Storage"),  # Charging: cb_palette[1]
            ("Edot_r_tot", -1, cb_palette[8], "From Storage"),  # Discharging: upgrade color
        ]
    elif with_storage:
        return [
            ("Edot_rem", 1, cb_palette[6], "Rest of Facility"),
            ("Edot_b", 1, cb_palette[0], blower_label),
            ("Edot_c", 1, cb_palette[2], "To Storage"),  # Charging: cb_palette[1]
        ]
    else:
        return [("Edot_rem", 1, cb_palette[6], "Rest of Facility")]


def figure_2_function(run_name, suffix="1.0__0", day="2022-07-01", npv_data=None, location_lookup=None, results=None):
    """
    Create a combined figure with technology profiles and NPV vs CAPEX plots for PSA with three upgrade keys.
    
    Args:
        run_name: Base run name
        suffix: Run suffix (default "base")
        day: Day for technology profiles (default "2022-07-01")
        npv_data: Optional DataFrame to use for the NPV bars plot (bottom row)
        location_lookup: Optional dict for x-axis label mapping
    """

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
    # Add subplot labels
    labels = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', '', '']
    for i, ax in enumerate(axes.flat):
        ax.text(-0.2, 1.05, labels[i], transform=ax.transAxes, fontsize=20, va='bottom', ha='left')
    
    # Get the first tariff key from the results
    tariff_key = list(results.keys())[0].split("___")[2] if results else "0.0__ebmud"
    first_config_key = "o2__psa__1.0__0"
    base_wwtp_key = first_config_key.split("___")[0]
    
    # Get upgrade keys from the results, but only for the current base_wwtp_key
    all_available_upgrades = []
    for config_key in results.keys():
        parts = config_key.split("___")
        if len(parts) >= 2:
            current_base_wwtp = parts[0]
            upgrade_key = parts[1]
            if current_base_wwtp == base_wwtp_key:
                all_available_upgrades.append(upgrade_key)
    
    upgrade_keys = []
    seen_upgrades = set()
    for upgrade_key in all_available_upgrades:
        if upgrade_key not in seen_upgrades and len(upgrade_keys) < 3:
            upgrade_keys.append(upgrade_key)
            seen_upgrades.add(upgrade_key)
    
    # Load data for each upgrade key - construct the correct data directory path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "output_data")
    
    all_power_data = []
    for col, upgrade_key in enumerate(upgrade_keys):
        upgrade_label_mappings = get_upgrade_label_mappings()
        base_upgrade_key = upgrade_key.split("__")[0] + "__" + upgrade_key.split("__")[1]
        upgrade_label = upgrade_label_mappings.get(base_upgrade_key, base_upgrade_key)
        
        # Load the configuration data
        config_key = get_config_name(base_wwtp_key, upgrade_key, tariff_key, suffix)
        with open(os.path.join(data_dir, run_name, f"{config_key}.pkl"), "rb") as f:
            config_data = pickle.load(f)
        all_results, best_design_key = get_design_data(config_data)     
        design_data = all_results[best_design_key]

        intermediate_dir = f"aeration_flexibility/output_data/{run_name}/intermediate/{config_key}"
        matching_files = glob.glob(os.path.join(intermediate_dir, f"{best_design_key}_*_{day}.pkl"))
        intermediate_file = matching_files[0]
        with open(intermediate_file, 'rb') as f:
            day_data = pickle.load(f)
            day_data = day_data["profile"]
        
        day_data["Ndot_b_aer_kg_hr"] = moles_to_mass(day_data["Ndot_b_aer"], M_O2)
        day_data["Ndot_target_kg_hr"] = moles_to_mass(day_data["Ndot_target"], M_O2)
        day_data["Edot_r_tot"] = day_data["Edot_r"] - day_data.get("Edot_r_o2", 0)
        for key in day_data:  # Round small values to zero
            day_data[key] = np.array([x if abs(x) >= 50 else 0 for x in day_data[key]])
        
        all_power_data.extend(day_data["Edot_t_baseline"] / 1000)  # Convert to MW
        all_power_data.extend(day_data["Edot_t_net"] / 1000)  # Convert to MW
        
        # TOP ROW: Technology profiles with modified power plotting   
        power_config = _figure_2_power_config(upgrade_key, base_wwtp_key, True, True)
        ax = axes[0, col]
        
        _figure_2_a_c(ax, day_data, power_config, True, upgrade_key)
        
        ax.set_title("")
        ax.set_xlim(0, 24)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["", "06:00", "12:00", "18:00", "24:00"], fontsize=16)                        
        ax.set_ylabel("Power (MW)", fontsize=20)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.set_xlabel("Time (hours)", fontsize=20)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

        # Add upgrade label inside the subplot
        ax.text(0.5, 0.95, upgrade_label, ha='center', va='top', 
                fontsize=16, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
        # Legend for baseline/optimized
        baseline_line = Line2D([0], [0], color='grey', linewidth=2, 
                                linestyle='--', label='Baseline')
        optimized_line = Line2D([0], [0], color='k', linewidth=2, 
                                linestyle='-', label='Optimized')
        axes[0, col].legend(handles=[baseline_line, optimized_line], 
                            loc='upper center', bbox_to_anchor=(0.5, 1.15),
                            fontsize=16, frameon=False, ncol=2)
    for col in range(len(upgrade_keys)):
        axes[0, col].set_ylim((min(all_power_data), max(all_power_data)*1.1))
    
    # MIDDLE ROW: NPV vs CAPEX plots
    for col, upgrade_key in enumerate(upgrade_keys):
        config_key = get_config_name(base_wwtp_key, upgrade_key, tariff_key, suffix)
        with open(os.path.join(data_dir, run_name, f"{config_key}.pkl"), "rb") as f:
            config_data = pickle.load(f)  
        all_results, best_design_key = get_design_data(config_data)
        design_data = all_results[best_design_key]
        
        ax = axes[1, col]
        capex_list = []
        npv_grid = []
        years_grid = []
        
        for design_key, design_data in all_results.items():
            metrics = design_data.get("metrics", {})
            capex = metrics.get("capex", None)
            annual_savings = metrics.get("opex", {}).get("savings", {})
            
            # For each lifetime (years)
            for years in range(5, 31):
                npv = calculate_itemized_npv(
                    annual_savings,
                    capex,
                    years=years
                )
                total_npv = npv["total"]
                capex_list.append(capex / 1e6)  # US$M
                years_grid.append(years)
                npv_grid.append(total_npv / 1e6)  # $M
        
        if capex_list:
            capex_arr = np.array(capex_list)
            years_arr = np.array(years_grid)
            npv_arr = np.array(npv_grid)
            
            # Create grid for interpolation with automatic CAPEX range based on actual data
            min_capex = max(0.1, np.nanmin(capex_arr))
            max_capex = min(10.0, np.nanmax(capex_arr))
            xi = np.linspace(min_capex, max_capex, 100)
            yi = np.linspace(5, 30, 100)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((capex_arr, years_arr), npv_arr, (xi, yi), method="cubic")
            setup_plot_style(ax, xlabel="Net Capex (US$M)", ylabel="Upgrade Lifetime (years)", fontsize=20)
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.set_ylim(5, 30)  # ylim from 5 to 30 for all bottom plots
            ax.set_xlim(min_capex, max_capex)
            ax.set_ylim(5, 25)
            ax.set_yticks([5, 10, 15, 20, 25, 30])
            ax.tick_params(axis='both', labelsize=24)
            
            cmap = get_custom_lcot_cmap(is_npv=True)
            vmin = np.nanmin(npv_arr)
            vmax = np.nanmax(npv_arr)
            
            # Handle edge cases for TwoSlopeNorm
            if np.isnan(vmin) or np.isnan(vmax):
                vmin, vmax = -15, 15
            elif vmin >= vmax:
                if vmin == vmax:  # idential values
                    abs_val = abs(vmin)
                    vmin = -max(abs_val, 1)  # Ensure at least some range
                    vmax = max(abs_val, 1)
                else:  # wrong order
                    vmin, vmax = vmax, vmin
            
            # For TwoSlopeNorm, we need vmin < vcenter < vmax
            if vmin > 0 and vmax > 0:  # all positive
                range_val = max(vmax - vmin, 1)  # Ensure at least some range
                vmin = -range_val
                vmax = range_val
            
            # Final validation: ensure vmin < 0 < vmax for TwoSlopeNorm
            if vmin >= 0:
                vmin = -max(abs(vmax), 1)
            if vmax <= 0:
                vmax = max(abs(vmin), 1)
            if vmin >= vmax:
                vmin, vmax = -1, 1
            
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) 
            im = ax.pcolormesh(xi, yi, zi, cmap=cmap, norm=norm, shading="auto")
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=14)
            im.set_clim(-30, 30)
            cbar.set_ticks([-30, -20, -10, 0, 10, 20, 30])
            cbar.set_ticklabels(['-30', '-20', '-10', '0', '10', '20', '30'])
            cbar.set_label("Net Present Value ($M)", fontsize=20)
            im.set_norm(norm)
            
            upgrade_label_mappings = get_upgrade_label_mappings()
            base_upgrade_key = upgrade_key.split("__")[0] + "__" + upgrade_key.split("__")[1]
            display_label = upgrade_label_mappings.get(base_upgrade_key, base_upgrade_key)
            ax.text(0.5, 0.95, display_label, ha='center', va='top', 
                    fontsize=16, transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Extract best upgrade data from bottom row and plot as dot on heatmap
            psa_data = npv_data[npv_data["Facility"].str.contains("psa", case=False)]
            full_upgrade_label = upgrade_key_labels[base_upgrade_key]
            upgrade_data = psa_data[psa_data["Upgrade Type"] == full_upgrade_label]
            best_upgrade = upgrade_data.loc[upgrade_data["npv"].idxmax()]
            best_capex = -best_upgrade["capex_npv"] / 1e6  # Convert back to positive capex in $M
            upgrade_color = cb_palette[upgrade_key_colors[full_upgrade_label]]
            ax.scatter(best_capex, 10, color=upgrade_color, s=800, marker="*", 
                        edgecolor="black", linewidth=0.5, zorder=10)  # using 10 years
            
            hours_list = []
            comp_ratio_list = []
            capex_list = []
            for design_key, design_data in all_results.items():
                metrics = design_data.get("metrics", {})
                capex = metrics.get("capex", None)
                if capex is None or not np.isfinite(capex):
                    continue
                
                # Parse design key for hours and compression ratio
                hours, comp_ratio = map(float, design_key.split("__"))
                hours_list.append(hours)
                comp_ratio_list.append(comp_ratio)
                capex_list.append(capex / 1e6)  # Convert to US$M
            
            base_name = f"{upgrade_key.replace('__', '_')}"
            output_dir = get_output_dir(f"{run_name}", "paper_figures")
            _figure_2_capex_scatter(hours_list, comp_ratio_list, capex_list, base_name, output_dir)
        
        setup_plot_style(ax, fontsize=14)
            
    # BOTTOM ROW: NPV bars facility plots with legend in third panel
    ax_combined = plt.subplot2grid((3, 3), (2, 0), colspan=2)

    # Hide FINAL 2 axes
    axes[2, 0].set_visible(False)
    axes[2, 1].set_visible(False)
    axes[2, 2].set_visible(False)

    plot_npv_bars(npv_data, ax=ax_combined)
    ax_combined.text(-0.15, 1.05, 'G)', transform=ax_combined.transAxes, fontsize=20, va='bottom', ha='left')
    if ax_combined.get_legend():
        ax_combined.get_legend().remove()

    # Legend in bottom right
    ax_legend = plt.subplot2grid((3, 3), (2, 2))    
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    legend_elements = []
    upgrade_types = npv_data["Upgrade Type"].unique()
    for upgrade in upgrade_types:
        color = cb_palette[upgrade_key_colors[upgrade]]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                            label=upgrade, edgecolor="black", linewidth=1.5))
    cost_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", 
                        linewidth=1.5, hatch=get_hatch_patterns()[label], label=label)
        for label in ["CAPEX", "Energy", "Demand", "Export", "H$_2$"]
    ]
    legend_elements.extend(cost_handles)
    legend_elements.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black", 
                    markersize=12, markeredgecolor="black", markeredgewidth=1.5, 
                    label="Total Delta LCOT (%)")
    )
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=20, frameon=False, ncol=1)
    # ax_legend.set_title("Legend", fontsize=18, fontweight='bold', pad=20)
    ax_legend.tick_params(axis='both', which='minor', length=0)
    ax_legend.axis('off')  # Remove plot area behind the legend

    plt.subplots_adjust(left=0.08, right=0.9, top=0.92, bottom=0.12, wspace=0.3, hspace=0.4)
    
    output_dir = get_output_dir(f"{run_name}", "paper_figures")
    save_path = os.path.join(output_dir, "psa_combined_figure.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to ensure a fresh start for the next plot


def figure_3_function(config, results):
    """Create tornado plot from results with stacked rows for each parameter group."""
    # Find baseline key (tariff with multipliers 1.0, 1.0, window 0)
    baseline_key = "0.0__svcw_d1.0_e1.0_w0"  # Can modify for EBMUD
    baseline_npv = results[baseline_key]

    # Map expected keys to group/row
    row_key_map = {
        "Demand Charges\n(-10%, +10%)": {
            "decrease": "0.0__svcw_d0.9_e1.0_w0",
            "increase": "0.0__svcw_d1.1_e1.0_w0",
        },
        "Energy Charges\n(-10%, +10%)": {
            "decrease": "0.0__svcw_d1.0_e0.9_w0",
            "increase": "0.0__svcw_d1.0_e1.1_w0",
        },
        "Peak Window\n(-0.5hr, +0.5hr)": {
            "decrease": "0.0__svcw_d1.0_e1.0_wneg0.25",
            "increase": "0.0__svcw_d1.0_e1.0_w0.25",
        },
        "Oxygen Requirement Ratio\n(-10%, +10%)": {
            "decrease": "0.9__0",
            "increase": "1.1__0",
        },
        # "Smoothing Factor\n(-2, +2)": {
        #     "decrease": "1.0__neg2",
        #     "increase": "1.0__2",
        # },
    }

    group_labels = list(row_key_map.keys())
    group_data = {label: {"decrease": None, "increase": None, "decrease_label": None, "increase_label": None} for label in group_labels}

    all_labels = config["parametrized_variations"]

    for group_name, keys in row_key_map.items():
        for var_type in ["decrease", "increase"]:
            key = keys[var_type]
            change = results[key] - baseline_npv
            pct_change = (change / abs(baseline_npv)) * 100
            group_data[group_name][var_type] = pct_change
            group_data[group_name][f"{var_type}_label"] = all_labels.get(key, key)

    # Print the constructed data for all rows
    for group_name, data in group_data.items():
        print(f"{group_name}: decrease={data['decrease']} ({data['decrease_label']}), increase={data['increase']} ({data['increase_label']})")

    # Plot
    plt.figure(figsize=(9, 6))
    y_pos = np.arange(len(group_labels))
    for i, group_name in enumerate(group_labels):
        data = group_data[group_name]
        plt.barh(
            y_pos[i],
            data["decrease"],
            color=cb_palette[3],
            alpha=0.7,
            label="Parameter Decrease" if i == 0 else "",
        )
        plt.text(
            data["decrease"],
            y_pos[i],
            f"{data['decrease']:+.1f}%",
            va="center",
            ha="right" if data["decrease"] < 0 else "left",
            color="black",
            fontsize=18,
        )
        plt.barh(
            y_pos[i],
            data["increase"],
            color=cb_palette[5],
            alpha=0.7,
            label="Parameter Increase" if i == 0 else "",
        )
        plt.text(
            data["increase"],
            y_pos[i],
            f"{data['increase']:+.1f}%",
            va="center",
            ha="right" if data["increase"] < 0 else "left",
            color="black",
            fontsize=18,
        )
    plt.yticks(y_pos, group_labels, fontsize=18)
    plt.gca().tick_params(axis="y", pad=30)
    plt.xlabel("Change in NPV (%)", fontsize=20)
    # plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.15),
        fontsize=18,
        frameon=False
    )
    plt.xticks(fontsize=18)
    plt.tight_layout()
    all_changes = [abs(data["decrease"]) for data in group_data.values() if data["decrease"] is not None]
    all_changes += [abs(data["increase"]) for data in group_data.values() if data["increase"] is not None]
    if all_changes:
        max_val = max(all_changes)
        buffer = max_val * 0.5
        plt.xlim(-max_val - buffer, max_val + buffer)
        plt.autoscale(False)
    output_dir = f'output_plots/{config["run_name"]}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f'{output_dir}/tornado_plot.png',
        bbox_inches="tight",
    )
    plt.close()


def figure_4_function(multipliers, hours_range, results, infeas, config):
    """Plot 2x3 subplot with top row showing heatmaps and bottom row showing NPV comparison."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # TOP ROW: Heatmaps for each upgrade technology
    technologies = [
        {"title": "Air", "data": results},
        {"title": "HPO from PSA", "data": results},
        {"title": "HPO from Cryo", "data": results}
    ]
    for idx, (ax, tech) in enumerate(zip(axes[0, :], technologies)):
        tech_results = tech["data"].copy()
        tech_infeas = infeas.copy()
            
        # Prepare data for this technology
        tech_hours_list = []
        tech_multipliers_list = []
        tech_lcots_list = []
        
        for i, multiplier in enumerate(multipliers):
            for j, hours in enumerate(hours_range):
                if not tech_infeas[i, j] and np.isfinite(tech_results[i, j]):
                    tech_hours_list.append(hours)
                    tech_multipliers_list.append(multiplier)
                    tech_lcots_list.append(tech_results[i, j])
        
        # Check if we have any data to plot
        if not tech_hours_list:
            print(f"Warning: No valid data points for {tech['title']} technology")
            continue
        
        # Check if all values are NaN
        if all(not np.isfinite(val) for val in tech_lcots_list):
            print(f"Warning: All LCOT values are NaN for {tech['title']} technology, setting to 0")
            tech_lcots_list = [0.0] * len(tech_lcots_list)
        
        # Check if we have enough data points for interpolation
        if len(tech_hours_list) < 3:
            print(f"Warning: Insufficient data points ({len(tech_hours_list)}) for {tech['title']} technology, setting to 0")
            # Create dummy data for plotting
            tech_hours_list = [12.0]  # Default to middle of range
            tech_multipliers_list = [1.0]  # Default to baseline
            tech_lcots_list = [0.0]  # Default to no change
        
        # Skip _plot_lcot_grid if all values are 0 (which means they were originally NaN)
        if all(val == 0.0 for val in tech_lcots_list):
            print(f"Warning: All values are 0 for {tech['title']} technology, skipping grid creation")
            continue
            
        xi, yi, zi = _plot_lcot_grid(tech_hours_list, tech_multipliers_list, tech_lcots_list)
    
        if xi is None or yi is None or zi is None:
            print(f"Warning: Could not create grid for {tech['title']} technology")
            continue

        # Determine min, max for LCOT normalization
        z_min = np.nanmin(zi)
        z_max = np.nanmax(zi)
        if not (np.isfinite(z_min) and np.isfinite(z_max)):
            print(f"Warning: All NaN values for {tech['title']} technology, setting to 0")
            # Create a grid of zeros instead of NaN
            zi = np.zeros_like(zi)
            z_min, z_max = 0, 0
            absmax = 1  # Set a small range to avoid normalization issues
        else:
            absmax = max(abs(z_min), abs(z_max))
        
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

        # Plot heatmap
        cmap = get_custom_lcot_cmap()
        im = ax.pcolormesh(xi, yi, zi, cmap=cmap, shading="auto", norm=norm)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, location='right', pad=0.05)
        cbar.set_label("Change in LCOT ($/m³)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        scatter = ax.scatter(
            tech_hours_list, tech_multipliers_list, c=tech_lcots_list, cmap=cmap, edgecolor="black", s=50, norm=norm
        )  # actual data points TODO: check if tech_hours_list should be hours

        setup_plot_style(ax, title=tech["title"], xlabel="Hours of O$_2$ Storage", ylabel="Compression Ratio")
        ax.set_ylim(min(tech_multipliers_list), max(tech_multipliers_list))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))        
        ax.set_xlabel("Hours of O₂ Storage", fontsize=18)
        y_labels = [f"{(tick-1)*100:.0f}%" for tick in ax.get_yticks()]
        ax.set_yticklabels(y_labels, fontsize=16)
        ax.set_ylabel("Increase in O₂ Required", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(0, 24)
        ax.set_title(ax.get_title(), fontsize=20)
        ax.text(-0.2, 1.05, chr(65 + idx) + ")", transform=ax.transAxes, fontsize=24, 
                va='bottom', ha='left')
    
    # BOTTOM ROW: Combined NPV comparison bar plot for 50% increase
    # Single subplot that spans all 3 columns
    fig.delaxes(axes[1, 1])
    fig.delaxes(axes[1, 2])
    bottom_ax = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    axes[1, 0].set_visible(False)
    
    # Extract NPV data for each upgrade technology
    upgrade_labels = ["Air", "HPO from PSA", "HPO from Cryo"]
    storage_npv = []
    counterfactual_npv = []
    mult_results = {}
    for config_key, config_data in config['results'].items():
        base_wwtp_key, upgrade_key, tariff_key, suffix = parse_config_key(config_key)
        parts = base_wwtp_key.split("__")
        if float(parts[2]) == 1.5:  # 50% increase
            mult_results[upgrade_key] = config_data
    for i, upgrade_type in enumerate(["none__gas_tank", "o2__psa", "o2__cryo"]):
        if upgrade_type not in mult_results:
            print(f"Warning: {upgrade_type} not found in mult_results, using dummy values")
            storage_npv.append(1.0)
            counterfactual_npv.append(1.0)
            continue
            
        all_results, best_design_key = get_design_data(mult_results[upgrade_type])
        design_data = all_results[best_design_key]
        storage_npv.append(design_data["npv"]["total"] / 1e6)  # Convert to $M
        counterfactual_npv.append(-design_data["npv"]["from capex savings"] / 1e6)  # Convert to $M
    
    x = np.arange(len(upgrade_labels))
    width = 0.35
    
    bar_colors = []
    for upgrade_label in upgrade_labels:
        if upgrade_label == "Air":
            color = cb_palette[1]
        elif "PSA" in upgrade_label:
            color = cb_palette[4]
        elif "Cryo" in upgrade_label:
            color = cb_palette[7]
        bar_colors.append(color)
    
    bars1 = bottom_ax.bar(x - width/2, storage_npv, width, label='Storage Upgrade NPV', 
                          color=bar_colors, alpha=0.8)
    
    bars2 = bottom_ax.bar(x + width/2, counterfactual_npv, width, label='Blower / HPO Expansion NPV', 
                          color=bar_colors, alpha=0.8, edgecolor='black', hatch='////') 
    for bar in bars1:
        height = bar.get_height()
        bottom_ax.text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:+.1f}M', ha='center', va='bottom' if height > 0 else 'top')
    for bar in bars2:
        height = bar.get_height()
        bottom_ax.text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:+.1f}M', ha='center', va='bottom' if height > 0 else 'top')
    
    bottom_ax.set_ylabel('Net Present Value (US$M)', fontsize=18)
    bottom_ax.set_title('NPV Comparison: Storage vs Counterfactual', fontsize=20)
    bottom_ax.set_xticks(x)
    bottom_ax.set_xticklabels(upgrade_labels, fontsize=16, rotation=45, ha='right')
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='Storage Upgrade NPV'),
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, hatch='////', label='Blower / HPO Expansion NPV')
    ]
    bottom_ax.legend(handles=legend_elements, fontsize=16)
    bottom_ax.grid(True, alpha=0.3)
    bottom_ax.text(-0.2, 1.05, "D)", transform=bottom_ax.transAxes, fontsize=24, 
                    va='bottom', ha='left')
    bottom_ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Save the plot
    output_dir = get_output_dir(config["run_name"], "paper_figures")
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.12, wspace=0.4, hspace=0.4)
    plt.savefig(os.path.join(output_dir, "summer_multiplier_heatmap.png"))
    plt.close()


def generate_plots(run_name="run_test", run_configs=None, location_lookup=None,
                   figure_2=False, figure_4=False, 
                   figure_3=False, si_storage=None, run_config=None):
        
    results = load_results_from_files(run_name)
    filtered_results = {}
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, tariff_key, suffix = parse_config_key(config_key)
        matching_run_config = None
        for run_config_key in run_configs.keys():
            if base_wwtp_key.startswith(
                run_config_key.split("__")[0]
                + "__"
                + run_config_key.split("__")[1]
            ):
                matching_run_config = run_config_key
                break
        if matching_run_config is None:
            continue
        matching_run_config_data = run_configs[matching_run_config]
        if upgrade_key in matching_run_config_data["upgrade_keys"] and tariff_key in (
            matching_run_config_data["tariff_configs"]
            if isinstance(matching_run_config_data["tariff_configs"], list)
            else [matching_run_config_data["tariff_configs"]]
        ):
            filtered_results[config_key] = config_data
    results = filtered_results
    output_dir = get_output_dir(run_name)

    get_output_dir(run_name, "mvn")
    plot_combined_metrics(results, run_name, output_dir, location_lookup=location_lookup)

    if si_storage:
        plot_storage_vs_bioreactor_volume(results, run_name, output_dir)
        si_cem_figure(results, run_name, output_dir)

    if figure_2:
        day = '2022-07-01'
        figure_2_function(run_name, suffix, day, npv_data=consolidate_npv_data(results, {}),
        location_lookup=location_lookup, results=results)

    if figure_4:
        multipliers = [config_item["multiplier"] for config_item in run_config["summer_config_template"]]
        o2_range = run_config["design_space"]["o2_range"]
        hours_range = np.linspace(o2_range[0], o2_range[1], 50)  # Use actual config range and increase resolution
        summer_results = np.full((len(multipliers), len(hours_range)), np.nan)
        infeas = np.zeros_like(summer_results, dtype=bool)
        
        # Process each configuration to extract delta LCOT values
        for config_key, config_data in results.items():
            base_wwtp_key, upgrade_key, tariff_key, suffix = parse_config_key(config_key)
            multiplier = float(split_key(base_wwtp_key)[2])
            if multiplier in multipliers:
                all_results, best_design_key = get_design_data(config_data)                                  
                design_data = all_results[best_design_key]
                hours, comp_ratio = map(float, best_design_key.split("__"))
                
                # Find indices with this multiplier
                mult_idx = multipliers.index(multiplier)
                # Find closest hours index instead of exact match TODO: compare with and without
                hours_idx = np.argmin(np.abs(hours_range - hours))
                
                # Calculate delta LCOT
                npv = design_data["npv"]["total"]
                if is_valid_npv(npv):
                    delta_lcot = npv_to_delta_lcot(npv)
                    summer_results[mult_idx, hours_idx] = delta_lcot
                else:
                    infeas[mult_idx, hours_idx] = True  # plot infeasible areas in black
        
        # Check if all values are NaN and set them to 0
        if np.all(np.isnan(summer_results)):
            print("Warning: All summer results are NaN, setting to 0")
            summer_results = np.zeros_like(summer_results)
        
        figure_4_function(multipliers, hours_range, summer_results, infeas, {"run_name": run_name, "results": results})
    
    # Generate tornado plot if requested
    if figure_3:
        tornado_results = {}
        summer_results = {}
        results = {}
        summer_configs = run_config["summer_configs"]
        for summer_config in summer_configs:
            suffix = get_summer_key(summer_config["multiplier"],summer_config["smoothing"])
            suffix_results = load_results_from_files(run_name)
            results.update(suffix_results)
        
        baseline_results = load_results_from_files(run_name)
        results.update(baseline_results)
        design_point = run_config["design_point"]
        design_key = f"{design_point[0]}__{design_point[1]}"
        tariff_keys_to_run = run_config.get("tariff_keys_to_run", [])
        for config_key, config_data in results.items():
            base_wwtp_key, upgrade_key, tariff_key, suffix = parse_config_key(config_key)
            all_results = config_data["all_results"]
            if design_key in all_results:
                npv = all_results[design_key]["npv"]["total"]
                # if is_valid_npv(npv):
                # Check if this is a tariff variation or summer config variation
                if tariff_key in tariff_keys_to_run:
                    # Check if this is actually a summer config variation by looking at base_wwtp_key
                    parts = base_wwtp_key.split("__")
                    if len(parts) >= 3 and parts[2] != "1.0":  # Has non-baseline summer config info
                        # This is a summer config variation
                        multiplier = float(parts[2])
                        smoothing = int(parts[3]) if len(parts) > 3 else 0
                        summer_key = get_summer_key(multiplier, smoothing)
                        summer_results[summer_key] = npv
                    elif len(parts) >= 4 and parts[3] != "0":  # Has non-baseline smoothing
                        # This is a smoothing variation (baseline multiplier but different smoothing)
                        multiplier = float(parts[2])
                        smoothing = int(parts[3])
                        summer_key = get_summer_key(multiplier, smoothing)
                        summer_results[summer_key] = npv
                    else:
                        # This is a tariff variation (baseline summer config with different tariff)
                        tornado_results[tariff_key] = npv
                else:
                    # This is a summer config variation (not in tariff_keys_to_run)
                    parts = base_wwtp_key.split("__")
                    multiplier = float(parts[2])
                    smoothing = int(parts[3]) if len(parts) > 3 else 0
                    summer_key = get_summer_key(multiplier, smoothing)
                    summer_results[summer_key] = npv

        all_tornado_results = {}
        all_labels = {}
        
        tariff_labels = get_tariff_key_labels(list(tornado_results.keys()))
        all_tornado_results.update(tornado_results)
        all_labels.update(tariff_labels)
        for key, npv in summer_results.items():
            all_tornado_results[key] = npv
            all_labels[key] = key
        
        tornado_config = {
            "run_name": run_config["run_name_from_file"],
            "parametrized_variations": all_labels
        }
        figure_3_function(tornado_config, all_tornado_results)
