import os.path
import re
import csv

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


#####setup
linestyle= ["-", ":", "--"]

# eval_episode_return of series6_hnppo_lever_2
returns = ["returns_opening_rates/run-a308f1ee series6_hnppo_lever_2 2022-09-18 14 07 37 algo=hnppo beta=0.0 clip-param=0.3 cuda-deterministic=yes env-nam_logs_hnppo_doorenv-v0_hnppo-lever-tag-eval_episode_rewards.csv",
           "returns_opening_rates/run-ce5e9c44 series6_hnppo_lever_2 2022-09-04 15 39 18 algo=hnppo beta=0.0 clip-param=0.3 cuda-deterministic=yes env-nam_logs_hnppo_doorenv-v0_hnppo-lever-tag-eval_episode_rewards.csv",
           "returns_opening_rates/run-f2aa0aa8 series6_hnppo_lever_2 2022-09-18 18 29 51 algo=hnppo beta=0.0 clip-param=0.3 cuda-deterministic=yes env-nam_logs_hnppo_doorenv-v0_hnppo-lever-tag-eval_episode_rewards.csv",
           ]

# opening rates of series6_hnppo_lever_2
opening_rates = ["returns_opening_rates/run-a308f1ee series6_hnppo_lever_2 2022-09-18 14 07 37 algo=hnppo beta=0.0 clip-param=0.3 cuda-deterministic=yes env-nam_logs_hnppo_doorenv-v0_hnppo-lever-tag-Opening_rate_per_update.csv",
                 "returns_opening_rates/run-ce5e9c44 series6_hnppo_lever_2 2022-09-04 15 39 18 algo=hnppo beta=0.0 clip-param=0.3 cuda-deterministic=yes env-nam_logs_hnppo_doorenv-v0_hnppo-lever-tag-Opening_rate_per_update.csv",
                 "returns_opening_rates/run-f2aa0aa8 series6_hnppo_lever_2 2022-09-18 18 29 51 algo=hnppo beta=0.0 clip-param=0.3 cuda-deterministic=yes env-nam_logs_hnppo_doorenv-v0_hnppo-lever-tag-Opening_rate_per_update.csv",
                 ]


def get_points(csvfile):
    plotpoints = []
    with open(csvfile) as f:
        r = csv.DictReader(f)
        for d in r:
            plotpoints.append((int(d['Step']), float(d['Value'])))
    return plotpoints


rt_plot = [list(zip(*get_points(r))) for r in returns]
op_plot = [list(zip(*get_points(o))) for o in opening_rates]


#####plot the data
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 2))
for i, plot in enumerate(rt_plot):
    ax1.plot(plot[0], plot[1], c=f"C{i}",)
    ax1.set_ylabel("Average Return")
    ax1.set_xlabel("Episode")
for i, plot in enumerate(op_plot):
    ax2.plot(plot[0], [p/100 for p in plot[1]], c=f"C{i}")
    ax2.set_ylabel("Opening rate")
    ax2.set_xlabel("Episode")
fig.tight_layout()
fig.savefig(f"return_opening_rate.png", bbox_inches="tight")
