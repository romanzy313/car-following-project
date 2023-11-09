# import json
# import matplotlib.pyplot as plt


# def plot_progress_histogram(data):
#     # Group data by ai_percent
#     grouped_data = {}
#     for entry in data:
#         ai_percent = entry["ai_percent"]
#         progress = entry["progress"]

#         if ai_percent not in grouped_data:
#             grouped_data[ai_percent] = []

#         grouped_data[ai_percent].append(progress)

#     # Plotting the histogram
#     fig, ax = plt.subplots()
#     for ai_percent, progress_list in grouped_data.items():
#         ax.hist(progress_list, bins=20, label=f"ai_percent: {ai_percent}", alpha=0.7)

#     ax.set_xlabel("Progress")
#     ax.set_ylabel("Frequency")
#     ax.legend(loc="upper right")
#     ax.set_title("Progress Histogram Grouped by AI Percent")

#     # Show the plot
#     plt.show()


# # Example usage with your provided dataset
# # dataset = [
# #     {"ai_percent": 0.8, "progress": 0.074},
# #     {"ai_percent": 0.8, "progress": 0.078},
# #     {"ai_percent": 0.8, "progress": 0.84},
# # ]
# with open("./results/test_aggregate.json", "r") as file:
#     dataset = json.load(file)
# plot_progress_histogram(dataset)

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_progress_lines(data):
    # Group data by ai_percent
    grouped_data = {}
    for entry in data:
        ai_percent = entry["ai_percent"]
        progress = entry["progress"]

        if ai_percent not in grouped_data:
            grouped_data[ai_percent] = []

        grouped_data[ai_percent].append(progress)

    # Plotting separate lines for each ai_percent
    for ai_percent, progress_list in grouped_data.items():
        sorted_progress = np.sort(progress_list)
        y_values = np.linspace(0, 1, len(sorted_progress))

        plt.plot(sorted_progress, y_values, label=f"ai_percent: {ai_percent}")

    plt.xlabel("Progress")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.title("Progress Line for Different AI Percent Scenarios")

    # Show the plot
    plt.show()


def plot_progress_curves(data):
    # Group data by ai_percent
    grouped_data = {}
    for entry in data:
        ai_percent = entry["ai_percent"]
        progress = entry["progress"]

        if ai_percent not in grouped_data:
            grouped_data[ai_percent] = []

        grouped_data[ai_percent].append(progress)

    # Plotting separate curves for each ai_percent
    for ai_percent, progress_list in grouped_data.items():
        plt.figure()
        axes = plt.gca()
        axes.set_xlim(0, 1)
        n, bins, _ = plt.hist(progress_list, bins=50, density=True, alpha=0.7)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, n, label=f"ai_percent: {ai_percent}", color="k")
        # plt.fill_between(bin_centers, 0, n, alpha=0.1, color="b")

        plt.xlabel("Progress")
        plt.ylabel("Density")
        # plt.legend()
        plt.title(f"Progress Curve for AI Percent: {ai_percent}")
        plt.savefig(f"./plots/progress_{ai_percent}.png")
        # plt.show()

    # Show the plots
    # plt.show()


# # Example usage with your provided dataset
# dataset = [
#     {"ai_percent": 0.8, "progress": 0.074},
#     {"ai_percent": 0.8, "progress": 0.078},
#     {"ai_percent": 0.8, "progress": 0.84},
#     {"ai_percent": 0.6, "progress": 0.5},
#     {"ai_percent": 0.6, "progress": 0.62},
#     {"ai_percent": 0.6, "progress": 0.72},
# ]
with open("./results/test_aggregate.json", "r") as file:
    dataset = json.load(file)

plot_progress_curves(dataset)
