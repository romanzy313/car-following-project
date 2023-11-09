import json
import pandas as pd
import matplotlib.pyplot as plt

with open("./results/test_aggregate.json", "r") as file:
    data = json.load(file)

##
df = pd.DataFrame(data)
##average veclocity with ai percent
average_velocity = df.groupby("ai_percent")["velocity_mean"].mean()
collision_counts = df.groupby("ai_percent")["collided"].sum()

# Plotting

fontsize = 18

plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
# plt.rc("xlabel", labelsize=20)
# plt.rc("ylabel", labelsize=20)

plt.figure(figsize=(9, 6))
plt.bar(
    average_velocity.index,
    average_velocity.values,
    width=0.1,
    align="center",
    alpha=0.7,
)
plt.xlabel("AI Population (%)", fontsize=fontsize)
plt.ylabel("Avg. Velocity (m/s)", fontsize=fontsize)
# plt.title("Average velocity of vechicels vs. AI Percent")
plt.xticks(average_velocity.index)
plt.grid(True)
# plt.show()
plt.savefig("./plots/Average velocity vs ai%.png", bbox_inches="tight")

plt.figure(figsize=(9, 6))
plt.bar(
    collision_counts.index,
    collision_counts.values,
    width=0.1,
    align="center",
    alpha=0.7,
)
plt.xlabel("AI Population (%)", fontsize=fontsize)
plt.ylabel("Number of Collisions", fontsize=fontsize)
# plt.title("Number of Collisions vs. AI Percent")
plt.xticks(collision_counts.index)
plt.grid(True)
# plt.show()
plt.savefig("./plots/collsion vs, ai %.png", bbox_inches="tight")
##
