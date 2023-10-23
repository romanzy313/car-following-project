import importlib


class Vehicle:
    def __init__(
        self,
        length: float,
        max_deceleration: float,
        max_acceleration: float,
    ):
        self.length = length
        self.max_deceleration = max_deceleration
        self.max_acceleration = max_acceleration
        # position and acceleration are just for visuals


def get_vehicle_from_name(class_name: str) -> type[Vehicle]:
    raise Exception("TODO, implement me")
    module = importlib.import_module(f"vehicles.{class_name}")
    class_ = getattr(module, "Definition")
    print("class is", class_)
    return class_
