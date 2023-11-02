def parse_length(val: str) -> float:
    """Converts any length unit into meters"""
    [value, unit] = val.split(" ")
    factor = 1

    if unit != "m":
        raise ValueError("length unit", unit, "not supported")

    return float(value) * factor


def parse_velocity(val: str) -> float:
    """Converts any velocity unit into meter per second"""
    [value, unit] = val.split(" ")
    factor = 1

    if unit == "m/s":
        pass
    elif unit == "km/h":
        factor = 1 / 3.6
    else:
        raise ValueError("velocity unit", unit, "not supported")

    return float(value) * factor


def parse_acceleration(val: str) -> float:
    """Converts any acceleration unit into m/s^2"""
    [value, unit] = val.split(" ")
    factor = 1

    if unit != "m/s^2":
        raise ValueError("acceleration unit", unit, "not supported")

    return float(value) * factor


def parse_time(val: str) -> float:
    """Converts any time unit into seconds"""
    [value, unit] = val.split(" ")
    factor = 1

    if unit == "s":
        pass
    elif unit == "ms":
        factor = 0.01
    else:
        raise ValueError("time unit", unit, "not supported")

    return float(value) * factor
