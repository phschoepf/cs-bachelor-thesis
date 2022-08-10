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
successful = results[(results['status'] != 'error') & (results['Episode_rewards'] >= -1e4)]  # promising trials
bad = results[(results['status'] != 'error') & (results['Episode_rewards'] < -1e4)]  # reward exploded into negative
error = results[results['status'] == 'error']  # not completed trials, usually due to grad explosion

bad['Episode_rewards'].clip(lower=-1e4, inplace=True)

fig, ax = plt.subplots()
# ax = Axes3D(fig, auto_add_to_figure=False)


def _scatter(ax, data, **kwargs):
    ax.scatter(data['lr'].apply(math.log10),
               data['clip-param'],
               # data['Episode_rewards'],
               **kwargs)


_scatter(ax, successful, alpha=1, cmap='jet', c=successful['Episode_rewards'])
_scatter(ax, bad, alpha=0.2, c='black')
_scatter(ax, error, marker='x')
ax.ticklabel_format(style='sci', scilimits=(0, 0))

fig.suptitle(os.path.splitext(os.path.basename(args.file))[0])
ax.set_xlabel('log(lr)')
ax.set_ylabel('clip-param')
# ax.set_zlabel('Episode rewards')

fig.add_axes(ax)

plt.ioff()
plt.show()
