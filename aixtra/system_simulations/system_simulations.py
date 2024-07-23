import numpy as np

def simulate(x_grid, simulation_name):
    '''Simulate the true values for the grid via the system simulation. Important note: Order of
    arguments must be identical to the order in the csv file.'''

    if simulation_name == "carnot":
        system_simulation = carnot_model
    elif simulation_name == "BopTest_TAir_ODE":
        system_simulation = boptest_delta_T_air_physical_approximation
    elif simulation_name == "BopTest_TAir_ODEel":
        system_simulation = boptest_delta_T_air_physical_approximation_elcontrol

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