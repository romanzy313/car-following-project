import argparse

from src.loader import loadScene

# from models.loader import

parser = argparse.ArgumentParser(
    prog="car-following-project",
    description="What the program does",
    epilog="Help text goes here",
)

# parser.add_argument("filename")  # positional argument
parser.add_argument("-s", "--scene", type=str, required=True)
parser.add_argument("-v", "--verbose", action="store_true")  # on/off flag

args = parser.parse_args()
# print(args)

loadScene(args.scene)
