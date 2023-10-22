class Vehicle:
    def __init__(
        self, id: str, length: float, max_deceleration: float, max_acceleration: float
    ):
        self.id = id
        self.length = length
        self.max_deceleration = max_deceleration
        self.max_acceleration = max_acceleration
        # position and acceleration are just for visuals

    def print_info(self):
        print(
            f"[Vehicle {self.id}]: Position {self.position}, Velocity {self.velocity}, Acceleration {self.acceleration}"
        )
