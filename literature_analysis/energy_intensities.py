import pandas as pd
import matplotlib.pyplot as plt
from ..oxygen_storage.helpers.parameters import cb_palette
import os

# cwd = os.getcwd()
# os.chdir(cwd)
df = pd.read_csv("energy_intensities.csv")
df.drop(columns=["Source"], inplace=True)

# Prepare data

df_n = df[df["Contaminant"] == "N"].drop(columns=["Contaminant", "Uses aeration?"])
df_n = df_n.groupby("Technology").mean().reset_index()
df_cod = df[df["Contaminant"] == "COD"].drop(columns=["Contaminant", "Uses aeration?"])
df_cod = df_cod.groupby("Technology").mean().reset_index()
df_n = df_n.sort_values(by="TRL", ascending=True)
df_cod = df_cod.sort_values(by="TRL", ascending=True)
df.set_index("Technology", inplace=True)


def color_x_ticks(ax, color):
    for label in ax.get_xticklabels():
        label.set_color(color)


def plot_colored_bars(
    df_cod,
    df_n,
    value_low,
    value_high,
    xlabel,
    filename,
    timing=False,
    xscale=None,
    x_tick_color=None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        df_cod["Technology"],
        df_cod[value_high] - df_cod[value_low],
        left=df_cod[value_low],
        color=cb_palette[2],
        alpha=0.6,
    )
    ax.barh(
        df_n["Technology"],
        df_n[value_high] - df_n[value_low],
        left=df_n[value_low],
        color=cb_palette[3],
        alpha=0.6,
    )
    ax.set_xlabel(xlabel)
    if xscale:
        ax.set_xscale(xscale)
    plt.tight_layout()

    # Color x-tick labels
    if x_tick_color is not None:
        color_x_ticks(ax, x_tick_color)

    # Remove legend
    if ax.get_legend():
        ax.get_legend().remove()

    # Optionally add timing axis
    if timing:
        ax2 = ax.twiny()
        ax2.set_xlim(0, 1)
        ax2.plot(
            df_cod["Oxygen Usage Timing"],
            df_cod["Technology"],
            "o",
            color="black",
        )
        ax2.plot(df_n["Oxygen Usage Timing"], df_n["Technology"], "o", color="black")
        ax2.set_xlabel(
            "Oxygen Usage Timing Indicator (0 - intermittent, 1 - continuous)"
        )

    plt.savefig(filename)
    plt.close()


def plot_energy_and_o2_overlay(df_cod, df_n, filename, x_tick_color=cb_palette[2]):
    fig, ax = plt.subplots(figsize=(7, 6))
    # Energy Intensity
    ax.barh(
        df_cod["Technology"],
        df_cod["Upper limit energy intensity (kwh/m3)"]
        - df_cod["Lower limit energy intensity (kwh/m3)"],
        left=df_cod["Lower limit energy intensity (kwh/m3)"],
        color=cb_palette[2],
        alpha=0.6,
    )
    ax.barh(
        df_n["Technology"],
        df_n["Upper limit energy intensity (kwh/m3)"]
        - df_n["Lower limit energy intensity (kwh/m3)"],
        left=df_n["Lower limit energy intensity (kwh/m3)"],
        color=cb_palette[3],
        alpha=0.6,
    )
    ax.set_xlabel("Energy Intensity (kWh/m3)")
    ax.set_xscale("log")
    plt.tight_layout()
    # Color x-tick labels
    color_x_ticks(ax, x_tick_color)
    if ax.get_legend():
        ax.get_legend().remove()
    ax2 = ax.twiny()
    # O2 Intensity
    ax2.barh(
        df_cod["Technology"],
        df_cod["Oxygen demand high (kg O2 / m3)"]
        - df_cod["Oxygen demand low (kg O2 / m3)"],
        left=df_cod["Oxygen demand low (kg O2 / m3)"],
        color=cb_palette[2],
        alpha=0.3,
    )
    ax2.barh(
        df_n["Technology"],
        df_n["Oxygen demand high (kg O2 / m3)"]
        - df_n["Oxygen demand low (kg O2 / m3)"],
        left=df_n["Oxygen demand low (kg O2 / m3)"],
        color=cb_palette[3],
        alpha=0.3,
    )
    ax2.set_xlabel("O2 Intensity (kg O2 delivered/m3)")
    ax2.set_xscale("log")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Plot 1: Energy Intensity (kWh/kg) - color x-ticks as COD
plot_colored_bars(
    df_cod,
    df_n,
    value_low="Lower limit energy intensity (kwh/kg)",
    value_high="Upper limit energy intensity (kwh/kg)",
    xlabel="Energy Intensity (kWh/kg)",
    filename="literature_analysis_outputs/energy_intensity_kg_colored_xticks.png",
    x_tick_color=cb_palette[2],
)

# Plot 2: Energy Intensity (kWh/m3) - color x-ticks as COD
plot_colored_bars(
    df_cod,
    df_n,
    value_low="Lower limit energy intensity (kwh/m3)",
    value_high="Upper limit energy intensity (kwh/m3)",
    xlabel="Energy Intensity (kWh/m3)",
    filename="literature_analysis_outputs/energy_intensity_m3_colored_xticks.png",
    xscale="log",
    x_tick_color=cb_palette[2],
)

# Plot 3: O2 Intensity (kg O2 / m3) + Timing - color x-ticks as COD
plot_colored_bars(
    df_cod,
    df_n,
    value_low="Oxygen demand low (kg O2 / m3)",
    value_high="Oxygen demand high (kg O2 / m3)",
    xlabel="O2 Intensity (kg O2 delivered/m3)",
    filename="literature_analysis_outputs/o2_intensity_m3_colored_xticks.png",
    timing=True,
    x_tick_color=cb_palette[2],
)

# Plot 4: Energy and O2 Intensity (m3) + Timing (overlayed axes) - color x-ticks as COD
plot_energy_and_o2_overlay(
    df_cod,
    df_n,
    "literature_analysis_outputs/energy_and_o2_intensity_m3_plus_timing_colored_xticks.png",
    x_tick_color=cb_palette[2],
)
plot_energy_and_o2_overlay(
    df_cod,
    df_n,
    "literature_analysis_outputs/energy_and_o2_intensity_m3_colored_xticks.png",
    x_tick_color=cb_palette[2],
)
