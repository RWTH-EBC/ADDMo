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