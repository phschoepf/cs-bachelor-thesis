import argparse
import os.path
import re

import matplotlib.pyplot as plt
import sqlite3
import yaml
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, nargs="+",
    default=["../../DoorGym/clmetrics/series7_config.yml",
             "../../DoorGym/clmetrics/series7_config_31415.yml",
             "../../DoorGym/clmetrics/series7_config_27182.yml"]
)
parser.add_argument("--ignore-tid", action="store_true", default=True)
args = parser.parse_args()

#####setup
linestyle= ["-", ":", "--"]
configs = []
for config in args.config:
    with open(config) as cf:
        configs.append(yaml.safe_load(cf))
db = sqlite3.connect("file:../../DoorGym/clmetrics/eval_results.sqlite?mode=ro",
                     uri=True,
                     detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
cur = db.cursor()
total_loads = 0

def get_highest_iters(config):
    return [int(re.match(r".*\.(\d+)\.pt$", run["checkpoint"]).group(1)) for run in config["runs"]]

normalize_to = list(map(max, zip(*[get_highest_iters(c) for c in configs])))

######load data from db
def get_plotdict(config):
    plot_dict = {}
    # iterate over tasks
    for task_id, eval_run in enumerate(config["runs"]):
        world = eval_run["world"]
        rates_outer = []
        # iterate over multiple runs (evaluate task for the run it was trained in and all future ones)
        run_id = eval_run["checkpoint"].split("/")[0]
        res = cur.execute("SELECT checkpoint, opening_rate "
                          "FROM evals "
                          "WHERE checkpoint like ? AND world like ? AND task_id = ?",
                          (f"%{run_id}%", f"%/{world}", -1 if args.ignore_tid else task_id))

        rates_inner = res.fetchall() or []
        global total_loads
        total_loads += len(rates_inner)
        rates_inner = [(int(re.match(r".*\.(\d+)\.pt$", chp).group(1)), rate) for chp, rate in rates_inner]
        try:
            rates_inner = [(chp / normalize_to[task_id] + task_id, rate) for chp, rate in rates_inner]
            rates_inner = sorted(rates_inner, key=lambda x: x[0])
            rates_inner.append((1 + task_id, rates_inner[-1][1]))
        except (ValueError, IndexError):
            # dummy opening rates if there is no data
            rates_inner = [(task_id, 0)]
        rates_outer.extend(rates_inner)
        plot_dict[world] = rates_outer
    return plot_dict


plot_dicts = [get_plotdict(c) for c in configs]


#####plot the data
fig, ax = plt.subplots(figsize=(10, 2))
for i, plot_dict in enumerate(plot_dicts):
    for j, (key, line) in enumerate(plot_dict.items()):
        line = list(zip(*line))
        ax.plot(line[0], line[1], ls=linestyle[i], c=f"C{j}", label=key.replace("_blue_floatinghook", "").replace("_fixed", ""))
    for start_value in range(1, len(configs[0]["runs"])):
        ax.axvline(start_value, color='black', linestyle='dashed', alpha=0.5)
    ax.set_xlim((0,len(configs[0]["runs"])))
    ax.set_ylim((-0.05, 1.05))
    ax.set_ylabel("Opening rate")
ax.title.set_text(f"PPO")
ax.tick_params(axis='x', color='white')
ax.set_xticks([i+0.5 for i in range(6)])
ax.set_xticklabels([f"Task {i}" for i in range(6)])
fig.tight_layout()
fig.savefig(f"cl_timeseries_{os.path.splitext(os.path.basename(args.config[0]))[0]}.png", bbox_inches="tight")
print(f"processed {total_loads} data points")
