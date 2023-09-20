class Vehicle:
    def __init__(
        self, id: str, length: float, max_deceleration: float, max_acceleration: float
    ):
        self.id = id
        self.length = length
        self.max_deceleration = max_deceleration
        self.max_acceleration = max_acceleration
