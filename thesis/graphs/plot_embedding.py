import numpy as np
import torch
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-policy-load", type=str, default="~/miniconda3/envs/doorgym/.guild/runs/3915ca6b405f453dbf539c2c0bbbd109/trained_models/hnppo/doorenv-v0_hnppo-lever_push_left/hnppo-lever_push_left.200.pt")
args = parser.parse_args()

actor_critic, _ = torch.load(args.pretrained_policy_load, map_location=torch.device("cpu"))
embeddings = actor_critic.base.hnet.task_embs.parameters()
emb_numpy = np.row_stack([emb.detach().numpy() for emb in embeddings])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(emb_numpy)

fig, ax = plt.subplots()
ax.plot(pca_result[:, 0], pca_result[:, 1], "o")
for i in range(pca_result.shape[0]):
    ax.annotate(str(i), (pca_result[i,0]+0.02, pca_result[i,1]))
fig.suptitle("Embedding PCA")
fig.savefig("../thesis/pca_test.png")
