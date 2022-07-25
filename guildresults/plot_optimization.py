import math
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()

results = pd.read_csv(args.file)
results = results[(results['Episode_rewards'] > -1e4)]  # remove runs with extremely low rewards

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
ax.scatter(results['lr'].apply(math.log10),
           results['clip-param'],
           results['Episode_rewards'],
           alpha=1, cmap='jet', c=results['Episode_rewards'])
ax.ticklabel_format(style='sci', scilimits=(0, 0))

fig.suptitle(os.path.splitext(os.path.basename(args.file))[0])
ax.set_xlabel('log(lr)')
ax.set_ylabel('clip-param')
ax.set_zlabel('Episode rewards')

fig.add_axes(ax)

plt.ioff()
plt.show()
