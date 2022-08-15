import math
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()

results = pd.read_csv(args.file)
successful = results[(results['status'] != 'error') & (results['Episode_rewards'] >= -1e4)]  # promising trials
bad = results[(results['status'] != 'error') & (results['Episode_rewards'] < -1e4)]  # reward exploded into negative
error = results[results['status'] == 'error']  # not completed trials, usually due to grad explosion

bad['Episode_rewards'].clip(lower=-1e4, inplace=True)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

#LUT = dict((k, v) for v, k in enumerate(set(successful['network-size'])))


def _scatter(ax, data, **kwargs):
    ax.scatter(data['lr'].apply(math.log10),
               data['clip-param'],
               data['entropy-coef'].apply(math.log10),
               **kwargs)

def _scatter2(ax, data, **kwargs):
    ax.scatter(data['network-width'],
               data['network-depth'],
               data['te-dim'],
               **kwargs)


_scatter(ax, successful, alpha=1, cmap='jet', c=successful['Episode_rewards'])
_scatter(ax, bad, alpha=0.2, c='black')
_scatter(ax, error, marker='x')

fig.suptitle(os.path.splitext(os.path.basename(args.file))[0])
ax.set_xlabel('log(lr)')
ax.set_ylabel('clip-param')
ax.set_zlabel('log(entropy)')


_scatter2(ax2, successful, alpha=1, cmap='jet', c=successful['Episode_rewards'])
_scatter2(ax2, bad, alpha=0.2, c='black')
_scatter2(ax2, error, marker='x')

ax2.set_xlabel('width')
ax2.set_ylabel('depth')
ax2.set_zlabel('te-dim')


plt.ioff()
plt.show()
