import pyomo.environ as pyo
import numpy as np
import json
import os
import sys
import pandas as pd
import time

from io import StringIO
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from helpers.parameters import *
from helpers.tariffs import *
from helpers.compressor_power import compressor_power, compressor_power_linear_coeff
from electric_emission_cost import costs

temp_stdout = StringIO()
sys.stdout = temp_stdout
sys.stdout = sys.__stdout__

HELPERS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
var_dict = json.load(open(os.path.join(HELPERS_DIR, "var_dict.json")))

profile_columns = [
    name
    for name, props in var_dict.items()
    if props.get("indexed") == True
    and props.get("length") == "var_length"
]


class O2Problem:

    def __init__(self, design_key, single_day_config=None, date=None, 
                 initial_SoS=None, data_wwtp=None, horizon_days=3):
        """Initialize the O2Problem with design parameters and constraints."""
        
        # Parse configuration
        self.base_key, self.upgrade_key, self.tariff_key = single_day_config.split(
            "___"
        )
        self.gas, self.o2_tech_base, summer_multiplier, summer_smoothing = split_key(
            self.base_key
        )

        self.o2_tech_upgrade = (
            self.upgrade_key.split("__")[0] if self.upgrade_key else None
        )
        self.o2_tech_for_storage = (
            self.o2_tech_upgrade
            if self.o2_tech_upgrade != "none"
            else self.o2_tech_base
        )
        self.storage_type = (
            self.upgrade_key.split("__")[1] if self.upgrade_key else None
        )
        self.nu_rec = self.upgrade_key.split("__")[2] if self.upgrade_key else None
        self.design_key = design_key
        self.date = date
        self.initial_SoS = initial_SoS

        self.has_elec = self.o2_tech_upgrade == "elec"
        self.has_surplus = self.o2_tech_upgrade == "none"
        self.is_cryo = self.o2_tech_base == "cryo"
        self.is_battery = self.storage_type == "battery"
        self.is_liquid_tank = self.storage_type == "liquid_tank"
        self.is_gas_tank = self.storage_type == "gas_tank"
        self.is_none = self.storage_type == "none"
        self.is_tank = self.is_liquid_tank or self.is_gas_tank

        self.horizon_days = horizon_days
        self.var_length = horizon_days*96

        self.cost_kwargs = {
            "desired_utility": "electric",
            "demand_scale_factor": self.var_length/(30*96),
            "resolution": '15m',
            "decompose_exports": True
        }

        self.m = pyo.ConcreteModel(
            name=f"{self.base_key}____{self.upgrade_key}___{self.design_key}___{date}"
        )

        self.var_dict = json.load(open(os.path.join(HELPERS_DIR, "var_dict.json")))

        self.profile_columns = []
        for category in self.var_dict["variables"].values():
            for var_name, props in category.items():
                if "indexed" not in props or props["indexed"] is True:
                    self.profile_columns.append(var_name)
        for param_name in self.var_dict["parameters"]["indexed"]:
            self.profile_columns.append(param_name)

    def get_remaining_new_param_vals(self, annual_param_limits):
        """Generate parameter values for a given design_key."""

        solar_multiplier = float(self.tariff_key.split("__", 1)[0])
        Hours_of_O2, compression_ratio = map(float, self.design_key.split("__"))

        # Storage parameters
        Ndot_b_max = annual_param_limits["Ndot_b_max"]
        Ndot_target_min = annual_param_limits["Ndot_target_min"]
        Ndot_target_mean = annual_param_limits["Ndot_target_mean"]
        E_max = annual_param_limits["Edot_b_mean"] * Hours_of_O2 if self.is_battery else 0

        # Calculate P_max then V_tank
        P_max = (
            compression_ratio * P_init_map[self.o2_tech_for_storage] + P_ATM
            if not self.is_liquid_tank
            else 1e12
        )  # upper limit ~7,000 psi
        V_tank = (Hours_of_O2 * Ndot_target_mean * R * T_room / (P_max * Pa_per_MPa * frac_o2_map[self.gas]))
        P_min = (P_ATM + P_AER) * 1.05
        N_min = (
            V_tank * (P_min * Pa_per_MPa) * frac_o2_map[self.gas] / (R * T_room)
            if not self.is_liquid_tank
            else 0
        )
        P_init = P_init_map[self.o2_tech_for_storage]
        N_max = Hours_of_O2 * Ndot_target_mean

        return {
            # Flow rate parameters
            "Ndot_b_max": Ndot_b_max,
            "Ndot_b_excess_max": Ndot_b_max if self.has_surplus else 0,
            "Ndot_b_min": (
                min(Ndot_target_min, (1 - cryo_turndown) * Ndot_b_max)
                if not self.is_cryo
                else min(Ndot_target_min, (1 - compressor_turndown) * Ndot_b_max)
            ),
            "Ndot_r_max": 0 if self.is_battery else annual_param_limits["Ndot_target_max"],
            "Ndot_c_max": 0 if self.is_battery else N_max/1,  # (limit / 1 hour)
            "unmet_o2_max": Ndot_b_max / 10,
            # Energy parameters
            "Edot_t_max": annual_param_limits["Edot_t_max"],
            "Edot_r_max": E_max / Hours_of_O2 if self.is_battery else 0,
            "Edot_c_max": annual_param_limits["Edot_c_max"],
            "E_max": E_max,
            # Pressure and volume parameters
            "P_min": P_min,
            "P_max": P_max,
            "P_init": P_init,
            "V_tank": V_tank,
            "N_min": N_min,
            "N_max": N_max,
            "N_max_h2": Ndot_target_mean * 2,
            # Efficiency parameters
            "eta_comp": eta,
            "eta_bat": 0.85,
            "nu_rec": 0,  # TODO: add option to change
            "l_o2": (0.05 if self.is_cryo else 0.02),  # 5% losses from liquid storage, 2% from gas
            "y_h2": 2 if self.has_elec else 0,
            "ei_o2_new": ei_o2_map[self.o2_tech_for_storage],
            "frac_o2": frac_o2_map[self.gas],
            # Solar parameters
            "solar_multiplier": solar_multiplier,
            "hourly_solar_multiplier": (
                hourly_solar_multiplier_default
                if solar_multiplier == 1.0
                else np.ones(self.var_length)
            ),
            "Edot_c_comp_h2_coeff": (
                compressor_power_linear_coeff(P_init, 35, rec=False)
                if self.has_elec
                else 0
            )
        }

    def create_vars_and_params(self, baseline_vals, new_vals):
        """Create variables and parameters for the model."""
        vals = {**baseline_vals, **new_vals}
        self.m.t = pyo.Set(initialize=range(int(self.var_length)))
        self.m.vp = pyo.Block()
        # self.vp = self.m.vp

        # Create parameters
        for param_name in self.var_dict["parameters"]["indexed"]:
            domain = pyo.Reals if self.var_dict["parameters"]["indexed"][param_name]["domain"] == "Reals" else pyo.NonNegativeReals
            param = pyo.Param(range(self.var_length), domain=domain, mutable=True, default=0)
            setattr(self.m.vp, param_name, param)
            
            # Handling for parameters that need to be extended for n-day optimization
            if param_name in ["hourly_solar_multiplier"] and isinstance(vals[param_name], np.ndarray) and len(vals[param_name]) == 96:
                # Repeat the 96-hour pattern n times
                extended_vals = np.tile(vals[param_name], self.horizon_days)
                for idx in range(self.var_length):
                    param[idx] = float(extended_vals[idx])
            else:
                for idx in range(self.var_length):
                    param[idx] = (
                        float(vals[param_name][idx])
                        if isinstance(vals[param_name], np.ndarray)
                        else float(vals[param_name].iloc[idx])
                    )

        for category_name, category in self.var_dict["parameters"]["non_indexed"].items():
            for param_name in category:
                domain = pyo.Reals if category[param_name]["domain"] == "Reals" else pyo.NonNegativeReals
                param = pyo.Param(domain=domain, mutable=True, default=0)
                setattr(self.m.vp, param_name, param)
                param.set_value(float(vals[param_name]))

        # Create variables
        def create_var(name, props):
            lb = 0 if props["bounds"][0] is None else props["bounds"][0]
            ub = None if props["bounds"][1] is None else props["bounds"][1]

            if isinstance(lb, str):
                lb = getattr(self.m.vp, lb)
            if isinstance(ub, str):
                ub = getattr(self.m.vp, ub)

            domain_str = props.get("domain", "NonNegativeReals")
            domain = pyo.Reals if domain_str == "Reals" else pyo.NonNegativeReals

            if props.get("indexed", True):
                var = pyo.Var(range(self.var_length), domain=domain, bounds=(lb, ub))
            else:
                var = pyo.Var(domain=domain, bounds=(lb, ub))

            setattr(self.m.vp, name, var)

        # Create base variables
        for name, props in self.var_dict["variables"]["base"].items():
            create_var(name, props)

        # Create storage variables based on type
        if self.is_battery:
            for name, props in self.var_dict["variables"]["energy_storage"].items():
                create_var(name, props)
        elif self.is_tank:
            for name, props in self.var_dict["variables"]["o2_storage"].items():
                create_var(name, props)
            if self.is_gas_tank:
                for name, props in self.var_dict["variables"]["gas_tank"].items():
                    create_var(name, props)

        return self.m

    def get_diagnostics(self):
        dsb = DiagnosticsToolbox(self.m)
        dsb.display_variables_at_or_outside_bounds()
        dsb.report_structural_issues()

    def initialize_vars(self):
        printed = False
        for t in range(self.var_length):
            if self.m.vp.Ndot_target[t].value < 1.0:
                if not printed:
                    print(f"Warning: Ndot_target <1.0 at time {t} for {self.date}. Setting to 1.0")
                    printed = True
                self.m.vp.Ndot_target[t].set_value(1.0)

            
            # 1. Initialize flow variables to match baseline blower operation
            blower_val = min(pyo.value(self.m.vp.Ndot_b_max), pyo.value(self.m.vp.Ndot_target[t]))
            self.m.vp.Ndot_b_aer[t].set_value(blower_val)
            self.m.vp.Ndot_b_excess[t].set_value(0)
            self.m.vp.Ndot_b[t].set_value(blower_val)
            
            # 2. Initialize storage flow variables to zero, or filling the gap of missing capacity
            if self.is_tank:
                gap = max(0, pyo.value(self.m.vp.Ndot_target[t]) - pyo.value(self.m.vp.Ndot_b_max))
                self.m.vp.Ndot_r[t].set_value(gap)
                self.m.vp.Ndot_c[t].set_value(0)

    def construct_constraints(self, charge_dict, prev_demand_dict):
        # Collect all constraint rules, then add constraints
        base_rules = {}
        self.m.base_const = pyo.Block()
        self._collect_base_constraints(base_rules)
        self._add_constraints(self.m.base_const, base_rules)

        if self.is_gas_tank or self.is_liquid_tank:
            o2_storage_rules = {}
            self.m.o2_storage_const = pyo.Block()
            self._collect_o2_storage_constraints(o2_storage_rules)
            self._add_constraints(self.m.o2_storage_const, o2_storage_rules)

        elif self.is_battery:
            energy_storage_rules = {}
            self.m.energy_storage_const = pyo.Block()
            self._collect_battery_constraints(energy_storage_rules)
            self._add_constraints(self.m.energy_storage_const, energy_storage_rules)

        cost_rules = {}
        self.m.cost_const = pyo.Block()
        self._collect_cost_constraints(charge_dict, prev_demand_dict, cost_rules)
        self._add_constraints(self.m.cost_const, cost_rules)

        return self.m

    def _add_constraints(self, block, rules):
        """Add constraints from a rules dictionary to a block."""
        for name, expr in rules.items():
            block.add_component(name, pyo.Constraint(expr=expr))

    def _collect_base_constraints(self, rules):

        # Create base expressions for use in base constraints
        base_expression_rules = {
            'Edot_b_comp': lambda m, t: self.m.vp.Edot_b_comp_coeff * self.m.vp.Ndot_b_aer[t] / self.m.vp.frac_o2,
            'Edot_b_gen': lambda m, t: self.m.vp.ei_o2_base * self.m.vp.Ndot_b_aer[t],
            'Edot_b_excess': lambda m, t: self.m.vp.Edot_b_comp_coeff * self.m.vp.Ndot_b_excess[t] / self.m.vp.frac_o2 + self.m.vp.ei_o2_base * self.m.vp.Ndot_b_excess[t]
        }
        for name, rule in base_expression_rules.items():
            setattr(self.m.vp, name, pyo.Expression(range(self.var_length), rule=rule))

        for t in range(self.var_length):
            next_t = t + 1
            rules[f"blower_power_{t}"] = self.m.vp.Edot_b[t] == self.m.vp.Edot_b_comp[t] + self.m.vp.Edot_b_gen[t]
            rules[f"excess_o2_{t}"] = self.m.vp.Ndot_b[t] == self.m.vp.Ndot_b_aer[t] + self.m.vp.Ndot_b_excess[t]
            rules[f"total_energy_balance_net_day1_t{t}"] = self.m.vp.Edot_t_net[t] == self.m.vp.Edot_t[t] - self.m.vp.Edot_c[t] * self.m.vp.solar_multiplier
            rules[f"solar_limit_{t}"] = self.m.vp.Edot_c[t] <= self.m.vp.Edot_t_max * self.m.vp.hourly_solar_multiplier[t] + 1e-3

            if next_t < self.var_length:
                rules[f"ramp_rate_up_{t}"] = self.m.vp.Edot_c[next_t] - self.m.vp.Edot_c[t] <= self.m.vp.Edot_t_max.value * 0.4

    def _collect_o2_storage_constraints(self, rules):

         # Create expressions for use in o2 storage constraints
        o2_storage_expression_rules = {
            'Edot_c_gen': lambda m, t: self.m.vp.ei_o2_new * self.m.vp.Ndot_c[t] + 1e-4,
            'Edot_c_comp': lambda m, t: compressor_power(self.m.vp.Ndot_c[t] / self.m.vp.frac_o2, self.m.vp.P_init, self.m.vp.P[t], self.m.vp.eta_comp, gas=self.gas, rec=False) if self.is_gas_tank else 0,
            'Edot_c_comp_h2': lambda m, t: self.m.vp.Ndot_c[t] * self.m.vp.y_h2 * self.m.vp.Edot_c_comp_h2_coeff if self.is_gas_tank else 0
        }
        for name, rule in o2_storage_expression_rules.items():
            setattr(self.m.vp, name, pyo.Expression(range(self.var_length), rule=rule))

        # Add initial storage state constraints
        if self.initial_SoS is not None:
            rules["initial_storage_moles"] = self.m.vp.N[0] == self.initial_SoS['N']
        else:
            rules["initial_storage_moles_min"] = self.m.vp.N[0] == self.m.vp.N_min

        # Timestep constraints
        for t in range(self.var_length):
            next_t = t + 1
            self._collect_o2_mass_balance_constraints(t, next_t, rules)
            if self.is_gas_tank and self.m.vp.V_tank.value > 0:
                self._collect_gas_tank_timestep_constraints(t, rules)
            elif self.is_liquid_tank:
                self._collect_liquid_tank_timestep_constraints(t, rules)

        # Add total_h2 constraint for gas tanks with electrolysis
        if self.is_gas_tank:
            rules["total_h2"] = self.m.vp.total_h2 == sum(self.m.vp.Ndot_c[t_] * self.m.vp.y_h2 for t_ in range(self.var_length)) / timestep_factor

    def _collect_o2_mass_balance_constraints(self, t, next_t, rules):
        # Only add mass balance constraint if next_t is within bounds (no wraparound)
        if next_t < self.var_length:
            rules[f"o2_storage_mass_balance_{t}"] = self.m.vp.N[next_t] == self.m.vp.N[t] + (self.m.vp.Ndot_c[t] + self.m.vp.Ndot_b_excess[t] - self.m.vp.Ndot_r[t]) / timestep_factor
        elif next_t == self.var_length:
            # Prevent "cheating" on the last timestep by fixing final storage state
            rules[f"final_storage_state_{t}"] = self.m.vp.N[t] == self.m.vp.N[t-1]
            # Prevent recovery flow at the final timestep
            rules[f"no_final_recovery_{t}"] = self.m.vp.Ndot_r[t] == 0

        rules[f"total_energy_balance_{t}"] = self.m.vp.Edot_t[t] == self.m.vp.Edot_rem[t] + self.m.vp.Edot_c[t] + self.m.vp.Edot_b[t] + self.m.vp.Edot_b_excess[t] + self.m.vp.Edot_r_o2[t] - self.m.vp.Edot_r[t]
        rules[f"o2_mass_balance_with_unmet_{t}"] = self.m.vp.Ndot_target[t] == self.m.vp.Ndot_b_aer[t] + (1 - self.m.vp.l_o2) * self.m.vp.Ndot_r[t] + self.m.vp.unmet_o2[t]
        rules[f"charge_energy_{t}"] = self.m.vp.Edot_c[t] == self.m.vp.Edot_c_gen[t] + self.m.vp.Edot_c_comp[t] + self.m.vp.Edot_c_comp_h2[t]

    def _collect_gas_tank_timestep_constraints(self, t, rules):
        rules[f"gas_storage_pressure_{t}"] = self.m.vp.P[t] == (self.m.vp.N[t] / self.m.vp.frac_o2) * R * T_room / (self.m.vp.V_tank * Pa_per_MPa)
        rules[f"Edot_r_{t}"] = self.m.vp.Edot_r[t] == compressor_power(self.m.vp.Ndot_r[t] / self.m.vp.frac_o2, self.m.vp.P_min, self.m.vp.P[t], self.m.vp.nu_rec, gas=self.gas, rec=True)
        rules[f"power_oxygen_rec_{t}"] = self.m.vp.Edot_r_o2[t] == 0

    def _collect_liquid_tank_timestep_constraints(self, t, rules):
        rules[f"power_oxygen_rec_{t}"] = self.m.vp.Edot_r_o2[t] == self.m.vp.ei_evap * self.m.vp.Ndot_r[t]
        rules[f"power_rec_{t}"] = self.m.vp.Edot_r[t] == 0

    def _collect_battery_constraints(self, rules):
        if self.initial_SoS is not None and 'E' in self.initial_SoS:
            rules["initial_storage_energy"] = self.m.vp.E[0] == self.initial_SoS['E']
        else:
            rules["initial_storage_energy_min"] = self.m.vp.E[0] == 0.0
        
        for t in range(self.var_length):
            rules[f"total_energy_balance_{t}"] = self.m.vp.Edot_t[t] == self.m.vp.Edot_c[t] + self.m.vp.Edot_b[t] + self.m.vp.Edot_b_excess[t] + self.m.vp.Edot_rem[t] - self.m.vp.Edot_r[t]
            rules[f"meet_demand_{t}"] = self.m.vp.Ndot_target[t] == self.m.vp.Ndot_b_aer[t]
            
            next_t = t + 1
            # Add battery energy balance constraint for points up to final point
            if next_t < self.var_length:
                rules[f"battery_energy_balance_{t}"] = self.m.vp.E[next_t] == self.m.vp.E[t] + (self.m.vp.Edot_c[t] * pyo.sqrt(self.m.vp.eta_bat) - self.m.vp.Edot_r[t] / pyo.sqrt(self.m.vp.eta_bat)) / timestep_factor
            elif next_t == self.var_length:  # fix final storage state - no "cheating"
                rules[f"final_storage_state_{t}"] = self.m.vp.E[t] == self.m.vp.E[t-1]

    def _collect_cost_constraints(self, charge_dict, prev_demand_dict, rules):
        """Collect cost constraint rules."""
        consumption_data_dict = {
            "electric": self.m.vp.Edot_t_net,
        }
        
        self.itemized_costs, self.m = costs.calculate_itemized_cost(
            charge_dict=charge_dict,
            consumption_data_dict=consumption_data_dict,
            model=self.m,
            prev_demand_dict=prev_demand_dict,
        )
        
        rules['total_cost'] = getattr(self.m.vp, 'tariff_cost') == self.itemized_costs["total"]

        # Add h2 value
        if self.is_gas_tank:
            rules["h2_value"] = self.m.vp.h2_value == moles_to_mass(self.m.vp.total_h2, M_H2) * price_h2_kg

    def add_cost_constraints(self, charge_dict, prev_demand_dict=None):
        """Add cost constraints to the model."""
        consumption_data_dict = {
            "electric": self.m.vp.Edot_t_net,
        }
        
        self.itemized_costs, self.m = costs.calculate_itemized_cost(
            charge_dict=charge_dict,
            consumption_data_dict=consumption_data_dict,
            model=self.m,
            prev_demand_dict=prev_demand_dict,
        )
        
        self.m.cost_const.add_component(
                'total cost',
                pyo.Constraint(expr=getattr(self.m.vp, 'tariff_cost') == self.itemized_costs["total"]),
            )

        # Add h2 value
        if self.is_gas_tank:
            self.m.cost_const.add_component(
                "h2_value",
                pyo.Constraint(
                    expr=self.m.vp.h2_value
                    == moles_to_mass(self.m.vp.total_h2, M_H2) * price_h2_kg
                ),
            )

    def _add_simul_penalty(self, obj_expr, charge_keys, discharge_keys):
        penalty_coeff=1e-5
        for t in range(self.var_length):
            charge_sum = sum(getattr(self.m.vp, key)[t] for key in charge_keys)
            discharge_sum = sum(getattr(self.m.vp, key)[t] for key in discharge_keys)
            obj_expr += penalty_coeff * (charge_sum + discharge_sum)
        return obj_expr
    
    def construct_objective(self):
        obj_expr = self.m.vp.tariff_cost
        
        if self.is_gas_tank:
            obj_expr -= self.m.vp.h2_value
        
        # Penalty for unmet oxygen demand
        unmet_penalty = sum(self.m.vp.unmet_o2[t] for t in range(self.var_length)) * 1e-1
        obj_expr += unmet_penalty
        
        # Penalty for simultaneous charge / discharge
        if 'tank' in self.storage_type:
            obj_expr = self._add_simul_penalty(obj_expr, charge_keys=['Ndot_c', 'Ndot_b_excess'], discharge_keys=['Ndot_r'])
        else:
            obj_expr = self._add_simul_penalty(obj_expr, charge_keys=['Edot_c'], discharge_keys=['Edot_r'])

        self.m.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
        return self.m

    def construct_problem(self, charge_dict, baseline_vals, new_param_vals, prev_demand_dict=None):
        self.create_vars_and_params(baseline_vals, new_param_vals)
        self.construct_constraints(charge_dict, prev_demand_dict)
        self.initialize_vars()
        self.construct_objective()
        return self.m

    def get_profile(self):
        sub_profile_columns = [column for column in self.profile_columns if column in self.m.vp.__dict__]

        if self.m.vp is None:
            print(f"{self.design_key} {self.date} Returning zeros because self.m.vp is None")
            return {key: np.zeros(self.var_length) for key in sub_profile_columns}

        if self.m.vp.Edot_t.extract_values() is None:
            print(f"{self.design_key} {self.date} Returning zeros because Edot_t.extract_values() is None")
            return {key: np.zeros(self.var_length) for key in sub_profile_columns}
        unmet_values = self.m.vp.unmet_o2.extract_values()
        if max(unmet_values.values()) > 10.0:
            print(f"{self.design_key} {self.date} Returning zeros because there is significant unmet o2")
            return {key: np.zeros(self.var_length) for key in sub_profile_columns}

        profile_dict = {
            column: pd.Series(
                {
                    k: v if v is not None else 0
                    for k, v in getattr(self.m.vp, column).extract_values().items()
                },
                index=getattr(self.m.vp, column).extract_values().keys(),
            )
            for column in sub_profile_columns
        }
        baseline_values = [pyo.value(self.m.vp.Edot_t_baseline[t]) for t in range(self.var_length)]
        values = [pyo.value(self.m.vp.Edot_t_net[t]) for t in range(self.var_length)]
        print(f"{self.design_key} {self.date} Edot_t_net: {round(min(values))}-{round(max(values))}, Edot_t_baseline: {round(min(baseline_values))}-{round(max(baseline_values))}, cost {round(self.m.vp.tariff_cost.value)}")
        return profile_dict

    def solve_optimization_day(self, tee=False):
        start_time = time.time()
        try:
            solver_config = {"max_iter": 5000, "tol": 1e-5}
            solver = pyo.SolverFactory("ipopt")
            solver.options.update(solver_config)
            results = solver.solve(self.m, tee=tee)
            solve_time = time.time() - start_time

            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                return self.get_profile()
            elif (
                results.solver.termination_condition
                == pyo.TerminationCondition.maxIterations
            ):
                print(
                    f"Warning: Maximum number of iterations exceeded for {self.m.name} after {solve_time:.3f}s."
                )
            else:
                print(f"  IPOPT failed after {solve_time:.3f}s. Status: {results.solver.termination_condition}")
                # self.get_diagnostics()
        except Exception as e:
            print(f"  Error solving {self.m.name}: {str(e)}")
            pass

        return {}

    def print_cost_values(self, charge_dict=None, prev_demand_dict=None):
        try:
            test_value = pyo.value(self.m.vp.Edot_t_net[0])
            test_value_2 = pyo.value(self.m.vp.tariff_cost)
            if test_value is None or test_value_2 is None:
                print("  Optimization did not solve - skipping cost calculation")
                return
        except (ValueError, AttributeError):
            print("  Optimization did not solve - skipping cost calculation")
            return

        baseline_consumption_data_dict = {
            "electric": np.array([pyo.value(getattr(self.m.vp, "Edot_t_baseline")[t]) for t in range(self.var_length)]),
        }
        
        baseline_itemized_costs, _ = costs.calculate_itemized_cost(
            charge_dict=charge_dict,
            consumption_data_dict=baseline_consumption_data_dict,
            prev_demand_dict=prev_demand_dict,
            **self.cost_kwargs
        )

        recalc_consumption_data_dict = {
            "electric": np.array([pyo.value(getattr(self.m.vp, "Edot_t_net")[t]) for t in range(self.var_length)]),
        }
        
        recalculated_cost, _ = costs.calculate_itemized_cost(
            charge_dict=charge_dict,
            consumption_data_dict=recalc_consumption_data_dict,
            prev_demand_dict=prev_demand_dict,
            **self.cost_kwargs
        )

        print(f"  tariff_cost cost: {round(pyo.value(self.m.vp.tariff_cost),0)}")
        print(f"  baseline cost: {round(baseline_itemized_costs['total'],0)}")
        print(f"  non-pyo-obj recalculated cost: {round(recalculated_cost['total'],0)}")
        print('pyo var itemized')
        for utility, utility_costs in self.itemized_costs.items():
            for charge_type, cost_value in utility_costs.items():
                if charge_type != "total":
                    print(f"    {utility}_{charge_type}: ${pyo.value(cost_value):,.2f}")

        sys.stderr.flush()
