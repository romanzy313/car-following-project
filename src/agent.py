# this is an agent class that will run at a fixed interval
# it will use agent-model like followahead.py to implement its brains

# The model will get position, velocity, and acceleration of vehicle in front
# and in the back. Agent outputs a value between -1 and 1 for throttle control

from typing import Any


class Agent:
    def __init__(self, name: str, config: Any) -> None:
        pass

    def process(self, dt: float):
        pass
