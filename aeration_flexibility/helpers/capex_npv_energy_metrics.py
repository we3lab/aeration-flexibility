import numpy as np
import math
import pandas as pd
import os
from helpers.parameters import *
from helpers.tariffs import *

cost_per_m3 = 10000  # material cost in $/m^3
labor_per_m2 = 100  # labor cost per m2 constructed
labor_per_tank = 2000  # labor cost per finished tank
markup = 0.5  # markup and overhed
Smax = 20000 * 0.00689476  # Maximum Allowable Stress (MPa)
E = 0.98  # Joint eff


def convert_to_2025_dollars(cost: float, from_year: int) -> float:
    """Convert a cost from a given year to 2025 dollars using inflation data."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inflation_df = pd.read_csv(os.path.join(project_root, "data", "inflation.csv"))

    yearly_inflation = (
        inflation_df.set_index("Year")["Average"].str.rstrip("%").astype(float) / 100
    )  # get value as float

    # Calculate cumulative forward inflation from source year to 2025
    cumulative_inflation = 1.0
    for year in range(from_year, 2025):
        if year in yearly_inflation.index:
            cumulative_inflation *= 1 + yearly_inflation[year]

    return cost * cumulative_inflation


def get_tank_cost(V_tank, P_max):
    """Calculate cost of gas storage tank system using ASME pressure vessel code.
    http://docs.codecalculation.com/mechanical/pressure-vessel/thickness-calculation.html

    Args:
        V_tank: Maximum volume in m3
        Pmax: Maximum pressure in MPa

    Returns:
        Tuple of (cost, number of tanks, wall thickness, aspect ratio)
    """
    if V_tank <= 0:
        print("V_tank is zero or negative - error")
        return np.nan, 0, 0, 0, 0

    tank_costs, ts, ns, ARs = [], [], [], []
    AR_options = np.linspace(4, 50, 100)
    n_options = np.arange(1, 40)

    for n in n_options:
        Vtank = V_tank / n
        for AR in AR_options:
            # Calculate tank radius
            r = (3 * Vtank / (2 * math.pi * (3 * AR - 1))) ** (1 / 3)

            # Calculate required thickness for circumferential stress (tc)
            if P_max <= 0.385 * Smax * E:
                tc = (P_max * r) / (Smax * E - 0.6 * P_max)
            else:
                tc = r * (math.exp(P_max / (Smax * E)) - 1)

            # Calculate required thickness for longitudinal stress (tl)
            if P_max <= 1.25 * Smax * E:
                tl = (P_max * r) / (2 * Smax * E + 0.4 * P_max)
            else:
                Z = P_max / (Smax * E) + 1
                tl = r * (math.sqrt(Z) - 1)

            # Required thickness is maximum of tc, tl, and 1.5 mmm
            t = max(tc, tl)

            if t > 0.02:  # maximum feasible thickness 2cm
                continue
            if t < 0.003:  # minimum thickness 1.5mm
                t = 0.003

            # Calculate MAWP for verification
            if t <= r / 2:
                MAWPc = (Smax * E * t) / (r + 0.6 * t)
                MAWPl = (2 * Smax * E * t) / (r - 0.4 * t)
            else:
                MAWPc = Smax * E * math.log((r + t) / r)
                Z = ((r + t) / r) ** 2
                MAWPl = Smax * E * (Z - 1)

            # Verify MAWP is sufficient
            if min(MAWPc, MAWPl) < P_max:
                continue

            # Calculate surface area and volume
            SA = 4 * AR * math.pi * r**2 + math.pi * (r**2)
            V_steel = SA * t

            # Calculate costs
            steel_cost = V_steel * cost_per_m3
            manufacturing_cost = SA * labor_per_m2 + n * labor_per_tank
            cost = (manufacturing_cost + steel_cost) * (1 + markup) * n

            tank_costs.append(cost)
            ts.append(t)
            ns.append(n)
            ARs.append(AR)

    # If no valid configurations were found
    if not tank_costs:
        # print(
        #     f"No valid tank configuration found for V_tank={V_tank:.2f} m3, P_max={P_max:.2f} MPa"
        # )
        return np.nan, 0, 0, 0

    # Find the configuration with minimum cost
    min_cost_idx = np.argmin(tank_costs)
    print(f"${tank_costs[min_cost_idx]} for {ns[min_cost_idx]} tanks total volume {V_tank}")
    return (
        tank_costs[min_cost_idx],
        ns[min_cost_idx],
        ts[min_cost_idx],
        ARs[min_cost_idx],
    )


def get_tank_cost_literature_by_vol(Vtank):
    # https://www.sciencedirect.com/science/article/pii/S2352152X23030438?via%3Dihub%27=#bb0240
    tank_cost = 4042 * Vtank ** 0.506
    return tank_cost


def get_tank_cost_literature(Vtank, Pmax):
    """Calculate tank cost based on literature model from Houssainy et al.

    Args:
        Pmax: Maximum pressure in Pa
        Vtank: Tank volume in m3

    Returns:
        Tuple of (cost, number of tanks, wall thickness, aspect ratio)

    Based on ASME code and vendor data from Houssainy et al.
    Loops through different configurations to find minimum cost.
    """
    if Vtank <= 0:
        print("Vtank is zero or negative - error")
        return np.nan, 0, 0, 0

    tank_costs, Ls, ns, ARs = [], [], [], []
    AR_options = np.linspace(4, 15, 20)  # Same AR range as get_tank_cost
    n_options = np.arange(1, 20)  # Same n range as get_tank_cost
    L_options = np.linspace(1.0, 3.0, 20)  # Length range from 1m to 3m

    for n in n_options:
        V_per_tank = Vtank / n
        for AR in AR_options:
            for L in L_options:
                D = L / AR  # Diameter in meters

                FPres = max(
                    ((Pmax) * D / (2 * (Smax * E - 0.6 * (Pmax))) + 0.00315) / 0.063,
                    1.0,
                )
                total_cost = 1000.0 * n * V_per_tank * FPres
                tank_costs.append(total_cost)
                ns.append(n)
                ARs.append(AR)
                Ls.append(L)
    min_cost_idx = np.argmin(tank_costs)

    return (
        tank_costs[min_cost_idx] * 10,
        ns[min_cost_idx],
        Ls[min_cost_idx],
        ARs[min_cost_idx],
    )


def get_h2_tank_cost(Nmax_kg):
    # https://www.hydrogen.energy.gov/docs/hydrogenprogramlibraries/pdfs/review23/st235_houchins_2023_p-pdf.pdf?Status=Master
    base_cost = 450 * Nmax_kg  # Cost in 2023 dollars
    return convert_to_2025_dollars(base_cost, 2023)  # Convert to 2025 dollars


def get_liquid_tank_cost(Nmax):
    """ Calculated liquid tank cost from first principles assuming cylindrical design"""
    
    Nmax_m3_liquid = moles_to_mass(Nmax, M_O2) / rho_hpo_liquid  # divide mass by density of liquid oxygen
    n_tanks = 1
    volume_per_tank = Nmax_m3_liquid

    # add more tanks until radius is less than 2 meters
    r = (3 * volume_per_tank / (4 * math.pi)) ** (1 / 3)
    while r > 2 and n_tanks < 100:  # Set a reasonable upper limit
        n_tanks += 1
        volume_per_tank = Nmax_m3_liquid / n_tanks
        r = (3 * volume_per_tank / (4 * math.pi)) ** (1 / 3)

    SA = 4 * math.pi * r**2

    V_steel = (
        SA * 0.01 * 4.0
    ) * 2.0  # thickness 10 mm, 4.0x markup since cylinder is conservative, double-walled insulation
    labor = labor_per_m2 * SA + labor_per_tank

    return (
        (V_steel * cost_per_m3 + labor) * (1 + markup) * n_tanks
    ) 


def battery_system_cost(capacity: float, power: float) -> float:
    """Calculate cost of battery system based on capacity and power rating.

    Based on NREL cost projections for utility-scale battery storage (2023 update).
    https://atb.nrel.gov/electricity/2024/commercial_battery_storage#capital_expenditures_(capex)

    Args:
        capacity: Battery capacity in kWh
        power: Battery power rating in kW

    Returns:
        Total battery system cost in USD
    """
    return convert_to_2025_dollars(capacity * 2749.83 + power * 687.46, 2023)


def mol_per_hr_to_ton_per_annum(flowrate: float) -> float:
    return (flowrate * 24 * 365) * (M_O2 / 1000000)


def psa_unit_cost(flowrate: float) -> float:
    """Calculate cost of PSA system per flowrate.

    Based on Tampa Bay Water Master Plan (Table 7.5-1).

    Args:
        flowrate: Flowrate in moles per hour

    Returns:
        PSA system cost in USD
    """
    return mol_per_hr_to_ton_per_annum(flowrate) * convert_to_2025_dollars(
        210, 2019
    )


def cryo_unit_cost(flowrate: float) -> float:
    """Calculate cost of cryogenic air separation system per flowrate.

    Based on Thunder Said Energy cryogenic air separation economics.

    Args:
        flowrate: Flowrate in moles per hour

    Returns:
        Cryogenic air separation system cost in USD
    """
    return mol_per_hr_to_ton_per_annum(flowrate) * convert_to_2025_dollars(
        200, 2020
    )  # TODO: check year


def elec_unit_cost(flow: float) -> float:
    """Calculate cost of electricity system per power rating.
    Based on EPRI TEA electrolysis CAPEX rates, cost in USD as a function of power rating in kW
    """
    power = flow * ei_elec_mol
    return power * convert_to_2025_dollars(2200, 2024)  # TODO: check year


def solar_unit_cost(power: float) -> float:
    """Calculate cost of solar system per power rating.

    Based on NREL solar installed system cost data.

    Args:
        power: Power rating in kW

    Returns:
        Solar system cost in USD
    """
    return power * convert_to_2025_dollars(1200, 2025)  # TODO: check year


def compressor_cost(power: float) -> float:
    """Calculate cost of rotary screw compressor.

    Based on Matche equipment cost data points.
    Regression fit: 1880 * (power ^ 0.671)

    Args:
        power: Power rating in kW

    Returns:
        Compressor cost in USD
    """
    power_hp = power * KW_TO_HP
    return convert_to_2025_dollars(1880 * power_hp**0.671, 2023)  # TODO: check year


def calculate_capex(sub_dict, storage_type, new_o2_supply_tech, base_wwtp_key, new_param_vals):
    """Calculate capital expenditure for a given configuration.

    Args:
        sub_dict: Dictionary containing parameter values and max values
        storage_type: Type of storage ('battery', 'gas_tank', etc.)
        new_o2_supply_tech: Oxygen supply technology ('psa', 'cryo', etc.)
        base_wwtp_key: Base WWTP configuration key
        new_param_vals: System new_param_vals dictionary

    Returns:
        Tuple of (total capex, tank metrics, capex components, counterfactual capex)
    """

    # Storage cost and tank metrics if applicable
    storage_cost = 0
    tank_metrics = {}
    if storage_type == "gas_tank":
        V_tank = sub_dict["param_vals"]["V_tank"]
        P_max = sub_dict["param_vals"]["P_max"] - P_ATM
        storage_cost, best_n, best_L, best_AR = get_tank_cost_literature(V_tank, P_max)
        tank_metrics = {"n_tanks": best_n, "length": best_L, "aspect_ratio": best_AR}
    elif storage_type == "liquid_tank":
        storage_cost = get_liquid_tank_cost(sub_dict["param_vals"]["N_max"])
    elif storage_type == "battery":
        E_max = sub_dict["param_vals"]["E_max"]
        max_Edot_c = sub_dict["max_values"]["Edot_c"] + 1e-4
        storage_cost = battery_system_cost(E_max, max_Edot_c)
    # Hydrogen storage cost
    h2_storage_cost = 0
    if "elec" in new_o2_supply_tech and "tank" in storage_type:
        N_max_h2 = sub_dict["param_vals"]["N_max"] * 2
        h2_storage_cost = get_h2_tank_cost(moles_to_mass(N_max_h2, M_H2))

    # O2 technology cost
    o2_tech_cost = 0
    if new_o2_supply_tech == "psa":
        o2_tech_cost = psa_unit_cost(sub_dict["max_values"]["Ndot_c"])
    elif new_o2_supply_tech == "cryo":
        o2_tech_cost = cryo_unit_cost(sub_dict["max_values"]["Ndot_c"])
    elif new_o2_supply_tech == "elec":
        o2_tech_cost = elec_unit_cost(sub_dict["max_values"]["Ndot_c"])
    elif new_o2_supply_tech == "compressor":
        o2_tech_cost = compressor_cost(sub_dict["max_values"]["Edot_c"])

     # Solar cost
    solar_cost = 0
    if sub_dict["billing_key"].startswith("1.0"):  # solar enabled
        solar_cost = solar_unit_cost(sub_dict["max_values"]["Edot_c"])

    # Total capex, adding additional costs
    base_equipment_cost = storage_cost + o2_tech_cost + h2_storage_cost + solar_cost
    if any(np.isnan(x) for x in [storage_cost, h2_storage_cost, o2_tech_cost, solar_cost]):
        total_capex, engineering_management_cost, construction_cost = np.nan, np.nan, np.nan
    else:
        if storage_type == "battery":  # smaller additional costs
            construction_cost = 0.1 * base_equipment_cost
            engineering_management_cost = construction_cost * 0.2
        elif new_o2_supply_tech == "elec":  # other costs included in capex function
            construction_cost = 0
            engineering_management_cost = 0
        else:
            construction_cost = 10000 + base_equipment_cost
            engineering_management_cost = construction_cost * 0.2
        total_capex = base_equipment_cost + engineering_management_cost + construction_cost

    # Counterfactual capex (cost of additional O2 capacity without storage)
    counterfactual_capex = 0
    o2_supply_tech = base_wwtp_key.split("__")[0]
    additional_o2_capacity = new_param_vals["Ndot_target_max"] - new_param_vals["Ndot_b_max"]
    if o2_supply_tech == "psa":
        counterfactual_capex = psa_unit_cost(additional_o2_capacity)
    elif o2_supply_tech == "cryo":
        counterfactual_capex = cryo_unit_cost(additional_o2_capacity)

    capex_components = {
        "h2_storage_cost": h2_storage_cost,
        "storage_cost": storage_cost,
        "o2_tech_cost": o2_tech_cost,
        "solar_cost": solar_cost,
        "construction_and_management_cost": (
            engineering_management_cost + construction_cost if not np.isnan(total_capex) else np.nan
        ),
    }
    return total_capex, tank_metrics, capex_components, counterfactual_capex
