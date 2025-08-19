#!/usr/bin/env python3
"""
Script to plot tank costs vs volume for different pressure ranges.
Creates a 1x3 subplot showing costs for 1 MPa, 10 MPa, and 100 MPa.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from ..aeration_flexibility.helpers.capex_npv_energy_metrics import get_tank_cost, get_tank_cost_literature, get_tank_cost_literature_by_vol

def plot_tank_costs():
    """Create tank cost plots for different pressure ranges."""
    
    # Define pressure ranges (in MPa)
    pressures = [1, 10, 100]
    pressure_labels = ['1 MPa', '10 MPa', '100 MPa']
    
    # Define volume range (in m³)
    volumes = np.logspace(0, 3, 50)  # 1 to 1000 m³
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (P_max, label) in enumerate(zip(pressures, pressure_labels)):
        ax = axes[i]
        
        # Calculate costs for all three functions
        costs_old = []
        costs_lit = []
        costs_lit_by_vol = []
        valid_volumes = []
        
        for V_tank in volumes:
            try:
                # Get costs from all functions
                cost_old, n_old, t_old, AR_old = get_tank_cost(V_tank, P_max)
                cost_lit, n_lit, L_lit, AR_lit = get_tank_cost_literature(V_tank, P_max * 1e6)  # Convert MPa to Pa
                cost_lit_by_vol = get_tank_cost_literature_by_vol(V_tank)
                
                # Include results if at least one literature function returns valid values
                if not np.isnan(cost_lit) or not np.isnan(cost_lit_by_vol):
                    costs_old.append(cost_old if not np.isnan(cost_old) else np.nan)
                    costs_lit.append(cost_lit if not np.isnan(cost_lit) else np.nan)
                    costs_lit_by_vol.append(cost_lit_by_vol)
                    valid_volumes.append(V_tank)
                    
            except Exception as e:
                print(f"Error calculating costs for V_tank={V_tank:.1f} m³, P_max={P_max} MPa: {e}")
                continue
        
        # Plot the results
        if valid_volumes:
            valid_volumes = np.array(valid_volumes)
            costs_old = np.array(costs_old)
            costs_lit = np.array(costs_lit)
            costs_lit_by_vol = np.array(costs_lit_by_vol)
            
            # Plot all three functions
            ax.loglog(valid_volumes, costs_old / 1000, 'b-', linewidth=2, label='ASME Code (get_tank_cost)')
            ax.loglog(valid_volumes, costs_lit / 1000, 'r--', linewidth=2, label='Literature Model (get_tank_cost_literature)')
            ax.loglog(valid_volumes, costs_lit_by_vol / 1000, 'g:', linewidth=2, label='Volume-based Model (get_tank_cost_literature_by_vol)')
            
            # Add grid and labels
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Tank Volume (m³)', fontsize=12)
            ax.set_ylabel('Cost (kUSD)', fontsize=12)
            ax.set_title(f'Tank Cost vs Volume at {label}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            
            # Add some statistics
            if len(valid_volumes) > 0:
                # Calculate ratios only for valid values at corresponding indices
                valid_mask = ~(np.isnan(costs_lit) | np.isnan(costs_old))
                if np.any(valid_mask):
                    valid_ratios = costs_lit[valid_mask] / costs_old[valid_mask]
                    cost_ratio = np.mean(valid_ratios)
                    ax.text(0.05, 0.95, f'Avg Cost Ratio (Lit/Old): {cost_ratio:.2f}', 
                           transform=ax.transAxes, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No valid configurations found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Tank Cost vs Volume at {label}', fontsize=14, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'output_plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tank_cost_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tank cost comparison plot saved to: {output_path}")
    
    # Also save as PDF for publication quality
    pdf_path = os.path.join(output_dir, 'tank_cost_comparison.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Tank cost comparison plot saved to: {pdf_path}")
    
    plt.show()

def print_cost_summary():
    """Print a summary of costs for specific volume-pressure combinations."""
    
    test_cases = [
        (10, 1),   # 10 m³ at 1 MPa
        (10, 10),  # 10 m³ at 10 MPa
        (10, 100), # 10 m³ at 100 MPa
        (100, 1),  # 100 m³ at 1 MPa
        (100, 10), # 100 m³ at 10 MPa
        (100, 100), # 100 m³ at 100 MPa
    ]
    
    print("\nTank Cost Summary:")
    print("=" * 120)
    print(f"{'Volume (m³)':<12} {'Pressure (MPa)':<15} {'ASME Cost (kUSD)':<18} {'Lit Cost (kUSD)':<18} {'Vol-based Cost (kUSD)':<20} {'Ratio (Lit/ASME)':<15}")
    print("-" * 120)
    
    for V_tank, P_max in test_cases:
        try:
            cost_old, n_old, t_old, AR_old = get_tank_cost(V_tank, P_max)
            cost_lit, n_lit, L_lit, AR_lit = get_tank_cost_literature(V_tank, P_max * 1e6)
            cost_lit_by_vol = get_tank_cost_literature_by_vol(V_tank)
            
            # Format output
            cost_old_str = f"{cost_old/1000:.1f}" if not np.isnan(cost_old) else "N/A"
            cost_lit_str = f"{cost_lit/1000:.1f}" if not np.isnan(cost_lit) else "N/A"
            cost_lit_by_vol_str = f"{cost_lit_by_vol/1000:.1f}"
            
            # Calculate ratio if both values are valid
            if not np.isnan(cost_old) and not np.isnan(cost_lit):
                ratio = cost_lit / cost_old
                ratio_str = f"{ratio:.2f}"
            else:
                ratio_str = "N/A"
            
            print(f"{V_tank:<12} {P_max:<15} {cost_old_str:<18} {cost_lit_str:<18} {cost_lit_by_vol_str:<20} {ratio_str:<15}")
                
        except Exception as e:
            print(f"{V_tank:<12} {P_max:<15} {'Error':<18} {'Error':<18} {'Error':<20} {'Error':<15}")

if __name__ == "__main__":
    print("Generating tank cost comparison plots...")
    plot_tank_costs()
    print_cost_summary() 