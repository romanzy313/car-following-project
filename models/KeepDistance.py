from src.model import Model


class Definition(Model):
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
