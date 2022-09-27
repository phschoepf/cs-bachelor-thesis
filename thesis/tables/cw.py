import numpy as np

"""Metrics for the Continuous World paper (training forward transfer, forgetting, timepoint accuracy).
CW metrics here are calculated between series5 (Continuous HNPPO run) and series6 (fresh HNPPO agent for each task) as
a reference. 
For CW, we limit the length of the runs to the shorter of the 2 (continuous or reference)."""

cw_metrics = [{'avg_performance': 0.7699999999999999,
               'forward_transfer': 0.3922165580259836,
               'forgetting': 0.003333333333333355},
              {'avg_performance': 0.57,
               'forward_transfer': 0.2301451533372553,
               'forgetting': 0.0050000000000000044},
              {'avg_performance': 0.715,
               'forward_transfer': 0.35661240507130637,
               'forgetting': 0.01833333333333333},
              ]

avg_dict = {}
for key in cw_metrics[0].keys():
    vals = [d[key] for d in cw_metrics]
    avg_dict[key] = (np.mean(vals), np.std(vals) / np.sqrt(len(vals)))
print("average cw:", avg_dict)
