from src.vehicle import Vehicle


class Scene:
    road_radius: float
    speed_limit: float
    framerate: int
    vehicles: dict[str, Vehicle] = {}

    def describe(self):
        print("total vehicles", len(self.vehicles))
