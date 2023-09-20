# reads the config and uses it for simulation
import json
from src.scene import Scene

from src.vehicle import Vehicle
from utils.unit_parsers import parse_acceleration, parse_length, parse_velocity


def loadScene(filepath: str) -> Scene:
    print("loading path", filepath)
    parsed = None
    with open(filepath) as file:
        parsed = json.loads(file.read())

    scene = Scene()

    # parse vehicles
    for data in parsed["vehicles"]:
        vehicle = Vehicle(
            id=data["id"],
            length=parse_length(data["length"]),
            max_acceleration=parse_acceleration(data["max_acceleration"]),
            max_deceleration=parse_acceleration(data["max_deceleration"]),
        )
        scene.vehicles[data["id"]] = vehicle

    # TODO parse agents

    # parse environment
    scene.framerate = int(parsed["environment"]["framerate"])
    scene.road_radius = parse_length(parsed["environment"]["road_radius"])
    scene.speed_limit = parse_velocity(parsed["environment"]["speed_limit"])

    scene.describe()
    return scene
