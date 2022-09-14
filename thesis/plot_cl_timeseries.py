import argparse
import re

import matplotlib.pyplot as plt
import sqlite3
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str)
parser.add_argument("--seed", choices=[1, 3415, 27182], type=int)
args = parser.parse_args()

#####setup
with open(args.config) as cf:
    config = yaml.safe_load(cf)
db = sqlite3.connect("file:../DoorGym/clmetrics/eval_results.sqlite?mode=ro",
                     uri=True,
                     detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
cur = db.cursor()

######load data from db
plot_dict = {}
# iterate over tasks
for task_id, eval_run in enumerate(config["runs"]):
    world = eval_run["world"]
    rates_outer = []
    # iterate over multiple runs (evaluate task for the run it was trained in and all future ones)
    for i, train_run in enumerate(config["runs"][task_id:]):
        run_id = train_run["checkpoint"].split("/")[0]
        res = cur.execute("SELECT checkpoint, opening_rate "
                          "FROM evals "
                          "WHERE checkpoint like ? AND world like ? AND task_id = ?",
                          (f"%{run_id}%", f"%/{world}", task_id))

        rates_inner = res.fetchall() or []
        rates_inner = [(int(re.match(r".*\.(\d+)\.pt$", chp).group(1)), rate) for chp, rate in rates_inner]
        highest_iter = max(r[0] for r in rates_inner)
        rates_inner = [(chp / highest_iter + task_id + i, rate) for chp, rate in rates_inner]
        rates_inner = sorted(rates_inner, key=lambda x: x[0])

        rates_outer.extend(rates_inner)
    plot_dict[world] = rates_outer


#####plot the data
fig, ax = plt.subplots(figsize=(10, 3.5))
for key, line in plot_dict.items():
    line = list(zip(*line))
    ax.plot(line[0], line[1], '-', label=key.replace("_blue_floatinghook", "").replace("_fixed", ""))
for start_value in range(1, len(config["runs"])):
    ax.axvline(start_value, color='black', linestyle='dashed', alpha=0.5)
ax.set_xlim((0,len(config["runs"])))
ax.set_ylabel("Opening rate")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27),
          ncol=len(plot_dict), fancybox=True, shadow=True)
ax.title.set_text(f"HNPPO timeseries\n{args.config}")
fig.tight_layout()
fig.savefig("cl_timeseries_test.png")
