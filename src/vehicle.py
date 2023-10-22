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
