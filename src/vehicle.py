import importlib


class Vehicle:
    def __init__(
        self,
        length: float,
        max_deceleration: float,  # unit m/s^2
        max_acceleration: float,  # unit m/s^2
        max_velocity: float,  # unit m/s
        display: str = "car",
        warn_on_limits=False,
    ):
        if max_deceleration < 0:
            raise Exception("max_deceleration must be positive!")

        self.length = length
        self.display = display
        self.max_deceleration = -max_deceleration
        self.max_acceleration = max_acceleration
        self.max_velocity = max_velocity
        self.warn_on_limits = warn_on_limits
        # position and acceleration are just for visuals

    def to_json(self):
        return {
            "length": self.length,
            "display": self.display,
            "max_deceleration": self.max_deceleration,
            "max_acceleration": self.max_acceleration,
        }

    def limit_acceleration(self, acceleration: float):
        filtered_acc = max(
            self.max_deceleration,
            min(self.max_acceleration, acceleration),
        )
        if filtered_acc != acceleration:
            if self.warn_on_limits:
                print(f"acceleration was limited from {acceleration} to {filtered_acc}")
            pass

        return filtered_acc

    def limit_velocity(self, velocity: float):
        filtered_velocity = max(
            0,
            min(self.max_velocity, velocity),
        )

        if filtered_velocity != velocity:
            if self.warn_on_limits:
                print(f"velocity was limited from {velocity} to {filtered_velocity}")
            pass

        return filtered_velocity
