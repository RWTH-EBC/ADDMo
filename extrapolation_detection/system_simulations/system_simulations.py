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
    return supply_temp + (t_amb - supply_temp) * (1 - 1 / (1 + p_el))

def boptest_delta_T_air_physical_approximation(t_amb, TAirRoom, rad_dir, u_hp) -> float:
    """
    This is a physical representation of the delta_T_air model for the BopTest.
    It is a simplified version of the model that is used in the BopTest.
    I think its from Stoffels Diss.

    Parameters
    ----------
    t_amb : float
        Ambient temperature in Kelvin or degrees Celsius.
    TAirRoom : float
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
    return (15000 / 35 * (t_amb - TAirRoom) + 24 * rad_dir + 15000 * u_hp) * 900 / 70476480