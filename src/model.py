from src.vehicle import Vehicle


class Model(Vehicle):
    def __init__(
        self,
        id: str,
        length: float,
        max_deceleration: float,
        max_acceleration: float,
        initial_position: float,
        inital_velocity: float,
    ):
        super().__init__(id, length, max_deceleration, max_acceleration)
        self.position = initial_position
        self.velocity = inital_velocity
        self.acceleration = 0

    def tick(self, delta_pos: float, delta_vel: float) -> float:
        """
        This is a standard model evaluation function.
        It takes in delta position and velocity to the vehicle ahead
        and returns acceleration
        """
        pass
