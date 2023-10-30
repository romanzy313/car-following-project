from src.vehicle import Vehicle
import importlib


class Model:
    def __init__(
        self,
        id: str,
        vehicle: Vehicle,
        initial_position: float,
        inital_velocity: float,
    ):
        self.id = id
        self.vehicle = vehicle
        self.position = initial_position
        self.velocity = inital_velocity

    def apply_acceleration(self, acceleration: float, dt: float):
        """
        This is how acceleration is applied.
        The accelerations are hard limited by min/max
        """
        filtered_acc = max(
            self.vehicle.max_deceleration,
            min(self.vehicle.max_acceleration, acceleration),
        )

        if filtered_acc != acceleration:
            print(f"acceleration was clamped from {acceleration} to {filtered_acc}")

        # TODO this is wrong? need to double check
        # this is fixed update, so it should be okay
        # https://www.youtube.com/watch?v=yGhfUcPjXuE
        self.velocity += filtered_acc * dt
        self.position += self.velocity * dt

        pass

    def to_json(self):
        return {"id": self.id, "position": self.position, "velocity": self.velocity}

    def print_state(self):
        print(f"[{id}] position {self.position} velocty {self.velocity}")

    # abstract functions

    def tick(self, delta_pos: float, delta_vel: float) -> float:
        """
        Abstract function
        This is a standard model evaluation function.
        It takes in delta position and velocity to the vehicle ahead
        and returns acceleration
        """
        raise Exception("Abstract me!")

    def inject_args(self, args):
        raise Exception("Abstract me!")


# https://stackoverflow.com/questions/4821104/dynamic-instantiation-from-string-name-of-a-class-in-dynamically-imported-module
def get_model_from_name(class_name: str) -> type[Model]:
    module = importlib.import_module(f"models.{class_name}")
    class_ = getattr(module, "Definition")
    print("class is", class_)
    return class_
