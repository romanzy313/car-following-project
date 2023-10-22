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

    def apply_acceleration(self, acceleration: float, dt: float):
        """
        This is how acceleration is applied.
        The accelerations are hard limited by min/max
        """
        filtered_acc = max(
            self.max_deceleration, min(self.max_acceleration, acceleration)
        )

        if filtered_acc != acceleration:
            print(f"acceleration was clamped from {acceleration} to {filtered_acc}")

        # TODO this is wrong? need to double check
        # this is fixed update, so it should be okay
        # https://www.youtube.com/watch?v=yGhfUcPjXuE
        self.velocity += filtered_acc * dt
        self.position += self.velocity * dt

        pass

    def json_state(self):
        return {"position": self.position, "velocity": self.velocity}

    def print_state(self):
        print(f"[{id}] position {self.position} velocty {self.velocity}")

    def tick(self, delta_pos: float, delta_vel: float) -> float:
        """
        Abstract function
        This is a standard model evaluation function.
        It takes in delta position and velocity to the vehicle ahead
        and returns acceleration
        """
        pass

    def inject_args():
        pass


class KeepDistanceModel(Model):
    keep_distance = 0

    def inject_args(self, args):
        self.keep_distance = args["keep_distance"]
        # print("injected keep distance", self.keep_distance)

    def tick(self, delta_pos: float, delta_vel: float) -> float:
        """
        This sample model tries to keep the same position as the vehicle in the front
        """

        return 0.1
        # return super().tick(delta_pos, delta_vel)
