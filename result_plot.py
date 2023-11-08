import json
import pandas as pd
import matplotlib.pyplot as plt

with open("test_aggregate.json", "r") as file:
    data = json.load(file)

##
df = pd.DataFrame(data)
##average veclocity with ai percent
average_velocity = df.groupby("ai_percent")["velocity_mean"].mean()
collision_counts = df.groupby("ai_percent")["collided"].sum()

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(
    average_velocity.index,
    average_velocity.values,
    width=0.1,
    align="center",
    alpha=0.7,
)
plt.xlabel("AI Percent")
plt.ylabel("Average velocity of vechicels(KM/H)")
plt.title("Average velocity of vechicels vs. AI Percent")
plt.xticks(average_velocity.index)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(
    collision_counts.index,
    collision_counts.values,
    width=0.1,
    align="center",
    alpha=0.7,
)
plt.xlabel("AI Percent")
plt.ylabel("Number of Collisions")
plt.title("Number of Collisions vs. AI Percent")
plt.xticks(collision_counts.index)
plt.grid(True)
plt.show()
##
