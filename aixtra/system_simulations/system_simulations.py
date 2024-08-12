import numpy as np

def system_factory(system_name: str):
    if system_name == "carnot":
        system_simulation = carnot_model
    elif system_name == "BopTest_TAir_ODE":
        system_simulation = boptest_delta_T_air_physical_approximation
    elif system_name == "BopTest_TAir_ODEel":
        system_simulation = boptest_delta_T_air_physical_approximation_elcontrol
    elif system_name == "bestest900_ODE":
        system_simulation = bestest900_ODE
    elif system_name == "bestest900_ODE_VL":
        system_simulation = bestest900_ODE_VL
    elif system_name == "bestest900_ODE_VL_COPcorr":
        system_simulation = bestest900_ODE_VL_COPcorr
    return system_simulation

def simulate(x_grid, simulation_name):
    '''Simulate the true values for the grid via the system simulation. Important note: Order of
    arguments must be identical to the order in the csv file.'''

    system_simulation = system_factory(simulation_name)

    y_grid = x_grid.apply(lambda row: system_simulation(*row), axis=1)

    return y_grid

def carnot_model(t_amb: float, p_el: float, supply_temp: float = 40) -> float:
    """Carnot model

    Parameters
    ----------
    t_amb: float
        Ambient temperature
    p_el: float
        Electrical power
    supply_temp: float
        Supply temperature

    Returns
    -------
    float
        Carnot model result
    """
    return p_el * (273.15 + supply_temp) / (supply_temp - t_amb)

def boptest_delta_T_air_physical_approximation(t_amb, rad_dir, u_hp, t_room) -> float:
    """
    This is a physical representation of the delta_T_air model for the BopTest.
    It is a simplified version of the model that is used in the BopTest.
    I think its from Stoffels Diss.

    Parameters
    ----------
    t_amb : float
        Ambient temperature in Kelvin or degrees Celsius.
    t_room : float
        Air room temperature in Kelvin or degrees Celsius.
    rad_dir : float
        Direct radiation in W/m^2.
    u_hp : float
        Heat pump modulation from 0 to 1.

    Returns
    -------
    float
        The calculated value based on the input parameters.

    """
    return (15000 / 35 * (t_amb - t_room) + 24 * rad_dir + 15000 * u_hp) * 900 / 70476480

def boptest_delta_T_air_physical_approximation_elcontrol(t_amb, rad_dir, u_hp, t_room) -> float:
    """
    This is a physical representation of the delta_T_air model for the BopTest.
    It is a simplified version of the model that is used in the BopTest.
    I think its from Stoffels Diss.

    Parameters
    ----------
    t_amb : float
        Ambient temperature in Kelvin or degrees Celsius.
    t_room : float
        Air room temperature in Kelvin or degrees Celsius.
    rad_dir : float
        Direct radiation in W/m^2.
    u_hp : float
        Heat pump modulation from 0 to 1.

    Returns
    -------
    float
        The calculated value based on the input parameters.

    """


    return (((15000 / 35) * (t_amb - t_room)) + (24 * rad_dir) + (15000 * u_hp * (((t_amb)/(273.15+35 - t_amb)) * 0.45/3) * (1 / (1 + np.exp(-(u_hp - 0.01) * 500))))) * 900 / 70476480

def bestest900_ODE(t_amb, rad_dir, u_hp, t_room) -> float:

    # hp stats
    COP_nominal = 3.33  # following boptest https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Fluid_HeatPumps.html#Buildings.Fluid.HeatPumps.ScrollWaterToWater
    exergetic_efficiency = 0.45 # following boptest https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Fluid_HeatPumps.html#Buildings.Fluid.HeatPumps.ScrollWaterToWater
    carnot = ((t_amb)/(t_room + 15 - t_amb))
    COP = carnot * exergetic_efficiency

    # heiz
    hp_nom = 15000
    hp_el_nom = hp_nom/COP_nominal
    heat_hp = hp_el_nom * COP * u_hp

    # building stats
    t_amb_auslegung = -15
    t_room_auslegung = 20
    transmission = ((hp_nom / (t_room_auslegung - t_amb_auslegung)) * (t_amb - t_room)) # annäherung der U-Werte durch Wärmepumpe Auslegung

    radiative = (24 * rad_dir)

    C_zone = 70476480
    time_step = 900
    capacity = (time_step / C_zone)

    delta_t = (transmission + radiative + heat_hp) * capacity
    return delta_t


def bestest900_ODE_VL(t_amb, rad_dir, u_hp, t_room) -> float:

    # hp stats
    COP_nominal = 3.33  # following boptest https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Fluid_HeatPumps.html#Buildings.Fluid.HeatPumps.ScrollWaterToWater
    exergetic_efficiency = 0.45 # following Stoffels Diss
    carnot = ((t_amb)/(t_room + (u_hp*15) + 5 - t_amb))
    COP = carnot * exergetic_efficiency

    # heiz
    hp_nom = 15000
    hp_el_nom = hp_nom/COP_nominal
    heat_hp = hp_el_nom * COP * u_hp

    # building stats
    t_amb_auslegung = -15
    t_room_auslegung = 20
    transmission = ((hp_nom / (t_room_auslegung - t_amb_auslegung)) * (t_amb - t_room)) # annäherung der U-Werte durch Wärmepumpe Auslegung

    radiative = (24 * rad_dir)

    C_zone = 70476480
    time_step = 900
    capacity = (time_step / C_zone)

    delta_t = (transmission + radiative + heat_hp) * capacity
    return delta_t

def bestest900_ODE_VL_COPcorr(t_amb, rad_dir, u_hp, t_room) -> float:

    # hp stats
    COP_nominal = 3.33  # following boptest https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Fluid_HeatPumps.html#Buildings.Fluid.HeatPumps.ScrollWaterToWater
    exergetic_efficiency = 0.45 # following Stoffels Diss
    COP_correction = ((-0.6*((u_hp-0.5)**2)) + 1.15) # Abgeleitet aus Diss von Vering
    carnot = ((t_amb)/(t_room + (u_hp*15) + 5 - t_amb))
    COP = carnot * exergetic_efficiency * COP_correction

    # heiz
    hp_nom = 15000
    hp_el_nom = hp_nom/COP_nominal
    # hp_el_nom = hp_el_nom * 2 # twice as big heat pump required for same heating power (due to on/off water pump and radiators instead of floor heating)
    heat_hp = hp_el_nom * COP * u_hp

    # building stats
    t_amb_auslegung = -15
    t_room_auslegung = 20
    transmission = ((hp_nom / (t_room_auslegung - t_amb_auslegung)) * (t_amb - t_room)) # annäherung der U-Werte durch Wärmepumpe Auslegung

    radiative = (24 * rad_dir)

    C_zone = 70476480
    time_step = 900
    capacity = (time_step / C_zone)

    delta_t = (transmission + radiative + heat_hp) * capacity
    return delta_t

def bestest900_ODE_bivalent(t_amb, rad_dir, p_el, t_room) -> float:

    # hp stats
    COP_nominal = 3.33  # following boptest https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Fluid_HeatPumps.html#Buildings.Fluid.HeatPumps.ScrollWaterToWater
    exergetic_efficiency = 0.3 # following boptest https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Fluid_HeatPumps.html#Buildings.Fluid.HeatPumps.ScrollWaterToWater
    # COP_correction = (-0.6*((u_hp-0.5)**2)) + 1.15 # Abgeleitet aus Diss von Vering
    carnot = t_amb/(273.15 + 35 - t_amb)
    COP = carnot * exergetic_efficiency

    # heiz und el stats
    hp_nom = 10000
    aux_nom = 5000
    heat_nom = 15000

    hp_el_nom = hp_nom/COP_nominal
    aux_el_nom = aux_nom/1
    el_nom = hp_el_nom + aux_el_nom # 8000
    hp_modulation = 0.2

    # heating
    abs_modulation_min = (hp_el_nom * hp_modulation) / el_nom
    abs_modulation_max = (hp_el_nom * 1) / el_nom

    p_el_to_20 = (1 / (1 + np.exp((p_el - abs_modulation_min) * 500)))
    p_el_from_20 = (1 / (1 + np.exp(-(p_el - abs_modulation_min) * 500)))
    p_el_to_100 = (1 / (1 + np.exp((p_el - abs_modulation_max) * 500)))
    p_el_from_100 = (1 / (1 + np.exp(-(p_el - abs_modulation_max) * 500)))


    heat_aux_low = el_nom * p_el_to_20 * p_el
    heat_aux_high= el_nom * p_el_from_100 * (p_el - abs_modulation_max)
    u_hp = p_el - (1/abs_modulation_max) # 0 bis 1 zwischen p_el=0 und p_el=abs_modulation_max
    heat_hp = hp_el_nom * COP * (p_el_from_20 * u_hp - p_el_from_100 * (u_hp- abs_modulation_max*(u_hp/p_el)))
    heater = heat_aux_low + heat_hp + heat_aux_high

    # building stats
    t_amb_auslegung = -15
    t_room_auslegung = 20
    transmission = ((heat_nom / (t_room_auslegung - t_amb_auslegung)) * (t_amb - t_room)) # annäherung der U-Werte durch Wärmepumpe Auslegung
    transmission = 428.57 * (t_amb - t_room)

    radiative = (24 * rad_dir)

    C_zone = 70476480
    time_step = 900
    capacity = (time_step / C_zone)
    capacity = 0.00001277


    delta_t = (transmission + radiative + heater) * capacity
    return delta_t



