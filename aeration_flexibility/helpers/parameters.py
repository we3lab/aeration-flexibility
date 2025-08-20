import numpy as np

# Timesteps
timestep = 15  # minutes
timestep_factor = 60 / timestep

# Constants
g = 9.8  # gravitational constant (m/s/s)
R = 8.3144621  # Gas constant, (J/K*mol)
M_O2 = 32  # molecular weight of O2 (g O2 / mol O2)
M_H2 = 2  # molecular weight of H2 (g H2 / mol H2)
n_air = 1.4  # ratio of specific heats for air
n_o2 = 1.4  # ratio of specific heats for oxygen
P_ATM = 0.1  # Atmospheric pressure (MPa)

MGD_TO_M3_PER_DAY = 3785.41  # 1 MGD = 3785.41 m^3/day
PLANT_CAPACITY_MGD = 13.5
HRT_DAYS = 0.25  # 12 hours
Q = PLANT_CAPACITY_MGD * MGD_TO_M3_PER_DAY  # m^3/day

# Conversions
scfm_to_m3hr = 1.699  # 60 min/hr * 0.0283 m3/scf
hp_to_kw = 1 / 1.34
m3_hr_to_mgd = 0.00634
KW_TO_HP = 1.34102209
MPa_to_psi = 145.038
Pa_per_MPa = 1e6

# BOD and nitrogen load and removal
f_BOD_primary = 0.35  # fraction of BOD remover in inlet works + primary clarifiers
f_BOD_aeration = (
    0.75  # fraction of BOD removed in aeration (not including primary removal)
)
BOD_in = 300  # mg/L BOD5
BOD_out = 20  # Effluent limits (mg/L BOD5)
epsilon_BOD = 1.1  # kg O2 required to oxidize 1 kg BOD
f_N_primary = 0.35  # fraction of N removed in inlet works + primary clarifiers
N_in = 50  # N load in influent (mg/L)
epsilon_N = 4.6  # kg O2 required to oxidize 1 kg NH3 to NO3 (Metcalf & Eddy 2012)

# Water properties
rho_ww = 1000  # density of wastewater (kg/m3)
rho_hpo_liquid = 1141 # density of liquid oxygen (kg/m3)

# Air and aeration system properties
rho_air = 1.204  # Density of air (kg/m3)
M_O2_air = 6.2 * 1000  # Molecular weight of O2 relative to  air (mg O2 / mol air)
M_air = 29.0 * 1000  # Average molecular weight of air (mg air / mol air)
frac_o2_air = 0.21  # Fraction of O2 in air
frac_o2_hpo = 0.95  # Fraction of O2 in high purity oxygen
alpha = 0.8  # correction factor to convert from clean water SOTE to wastewater
beta = 0.9  # assumption for any additional losses or fouling of diffusers
SOTE = {"Disc": 0.7, "Invent": 0.8}  # standard oxygen transfer efficiency (%)

o2_multiplier_map = {"air": 1, 
                     "o2": 0.7 / 0.6 / SOTE["Invent"]}
nu = {
    "Turbo": 0.75,
    "Centrifugal": 0.65,
    "RotaryLobe": 0.55,
}  # blower + motor operating efficiency

eta = 0.75
nu_comp = 0.70  # for rotary screw compressor, 0.65-0.7
polytropic_efficiency = 0.8
pressure_psi = 15  # Blower outlet pressure in psi
pressure_pa = 103421 * pressure_psi  # Blower outlet pressure in Pascals
P_AER = 0.1  # Pressure required to operate aeration system (MPa)
T_tank = 25 + 298  # temperature in kelvin
T_room = 20 + 298  # temperature in kelvin
P_MAX_H2 = 35.0  # MPa

# Finances
payback_period = 30  # payback period in years
payback_period_days = payback_period * 365  # days during payback period
inflation_rate = 0.01
days_per_month = 30
operating_rate = 0.01

# Energy intensities
ei_psa_kg = 0.79  # kWh/kg O2
ei_cryo_kg = 0.39  # kWh/kg O2
ei_elec_kg = 13.9  # kWh/kg O2
ei_elec_mol = ei_elec_kg / 1000 * M_O2  # kwh/mol O2
ei_psa_mol = ei_psa_kg / 1000 * M_O2  # kWh/mol O2
ei_cryo_mol = ei_cryo_kg / 1000 * M_O2  # kWh/mol O2

cryo_turndown = 0.1
compressor_turndown = 0.5

# 14.83 KW/MGD  from Howard Curren plant https://www.tampa.gov/sites/default/files/content/files/migrated/hfc_awtp_-_master_plan_-_volume_1_0.pdf

# Although VPSA systems tend to have
# higher capital costs, they can be turned down to 50% of total capacity which provides
# opportunity for operational savings. The VPSA System can also be automated to maintain
# dissolved oxygen set point in the reactors, allowing for more targeted control of the oxygen
# generation with the demand in the system. In general, electricity usage can be reduced by 30 to
# 70% as compared to a Cryogenic system.

price_h2_kg = 5.0  # $/kg
price_h2_mol = price_h2_kg * (M_H2 / 1000)  # $/kg * (g/mol * kg/g) = $/mol

energy_h2_mol = 0  # 154*0.000277778          # kWh/mol energy content https://www.nrel.gov/docs/fy10osti/47302.pdf for efficient fuel cell

# Mappings
frac_o2_map = {"air": frac_o2_air, 
               "o2": frac_o2_hpo
               }

n_map = {"air": n_air, 
        "o2": n_o2,
        }

ei_o2_map = {
    "psa": ei_psa_mol,
    "cryo": ei_cryo_mol,
    "elec": ei_elec_mol,
    "compressor": 0,
}

ei_evap_kj = 3.4099  # kJ/mol
ei_evap_kwh = ei_evap_kj / 1000 / 3600  # kWh/mol

ei_evap_map = {
    "psa": 0,
    "cryo": ei_evap_kwh,
    "elec": 0,
    "compressor": 0,
}

P_init_map = {
    "psa": P_ATM * 1.1,
    "cryo": P_ATM,
    "elec": P_ATM * 1.5,
    "compressor": P_ATM,
    "none": P_ATM,  # TODO: remove
}

l_o2 = 0.02

hourly_solar_multiplier_default = np.zeros(96)
for i in range(96):  # duck curve for solar
    hour = i / 4
    if 6 <= hour < 18:  # Daylight hours (6am to 6pm), bell curve peaking at noon
        hourly_solar_multiplier_default[i] = np.sin((hour - 6) * np.pi / 12)

desc_units = {
    "Total Baseline Power": "(kWh/day)",
    "Total Optimized Power": "(kWh/day)",
    "Maximum Baseline Power": "(kW)",
    "Round Trip Efficiency": "",
    "Energy Capacity": "(kWh)",
    "Normalized Energy Capacity": "(kWh/kWh)",
    "Power Capacity": "(kW)",
    "Normalized Power Capacity": "(kW/kW)",
}


def volume_to_moles_stp(volume):
    """
    Calculate moles of air based on ideal gas law

    Arguments:
        Volume (m3) or flowrate (m3/hr)

    Returns:
        Moles (mol) of flowrate (mol/hr)
    """
    temperature = T_room
    moles = P_ATM * Pa_per_MPa * volume / (R * temperature)
    return moles

print('volume to moles STP for 1 m3')
print(volume_to_moles_stp(1))


def moles_to_mass(moles, M):
    """
    Calculate mass of gas based on ideal gas law
    Input moles (or moles/hr) in mol
    Returns mass (or mass/hr) in kg
    """
    mass = moles * M / 1000
    return mass


cb_palette = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
# Summer tariffs: May 1 through October 30
# Winter tariffs: November 1 through April 30
# Dry season NPDES: From May 1 through September 30,
# Wet season NPDES: From October 1 through April 30,


def get_all_days_in_month(year, month):
    year = int(year)
    month = int(month)
    days_in_month = (
        31
        if month in [1, 3, 5, 7, 8, 10, 12]
        else (
            30
            if month in [4, 6, 9, 11]
            else (
                29
                if month == 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
                else 28
            )
        )
    )
    return [f"{year}-{month:02d}-{day:02d}" for day in range(1, days_in_month + 1)]


def split_key(key):
    return key.split("__")


def get_summer_key(multiplier, smoothing):
    if smoothing < 0:
        summer_key = f"{multiplier}__neg{abs(smoothing)}"
    else:
        summer_key = f"{multiplier}__{smoothing}"
    return summer_key


def get_config_name(base_wwtp_key, upgrade_key, tariff_key, suffix):
    return f"{base_wwtp_key}___{upgrade_key}___{tariff_key}___{suffix}"
