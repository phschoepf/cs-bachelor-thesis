import numpy as np
import sys

import torch
from tqdm import trange
from clfd.imitation_cl.model.hypernetwork import *
from split_mnist import SplitMNIST

if __name__ == "__main__":
    task_id: int = sys.argv[1] if len(sys.argv) > 1 else 0
    config = {"beta": 5e-3,
              "lr": 1e-4,
              }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hnet = HyperNetwork(layers=[2048, 512, 128],
                        te_dim=5,
                        target_shapes=TargetNetwork.weight_shapes(n_in=28 * 28, n_out=10, hidden_layers=[512, 64]),
                        device=device).to(device)

    tnet = TargetNetwork(n_in=28 * 28,
                         n_out=10,
                         hidden_layers=[512, 64],
                         no_weights=True,
                         bn_track_stats=False,
                         device=device).to(device)

    hnet.train()
    tnet.train()
    hnet.gen_new_task_emb()

    if config["beta"] > 0:
        targets = get_current_targets(task_id, hnet)
    else:
        targets = None

    # Trainable weights and biases of the hnet
    regularized_params = list(hnet.theta)

    # For optimizing the weights and biases of the hnet
    theta_optimizer = optim.Adam(regularized_params, lr=config["lr"])

    # For optimizing the task embedding for the current task.
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    emb_optimizer = optim.Adam([hnet.get_task_emb(task_id)], lr=config["lr"])

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and config["beta"] > 0

    criterion = torch.nn.MSELoss()
    # Start training iterations
    for epoch in trange(5):

        running_loss = 0.0
        for i, (x, y) in enumerate(DataLoader(SplitMNIST("~/Desktop/schoepf-bachelor-thesis/cmnist/data",
                                                         classes=[0, 1],
                                                         transform=lambda img: np.asarray(img, dtype=np.float32).flatten()),
                                              batch_size=16,
                                              shuffle=True)):

            # make a one-hot tensor from the targets
            y = torch.nn.functional.one_hot(y, num_classes=10).float().to(device)

            x = x.to(device)
            ### Train theta and task embedding
            theta_optimizer.zero_grad()
            emb_optimizer.zero_grad()

            # Populate weights of tnet with outputs of hnet
            weights = hnet.forward(task_id)
            tnet.set_weights(weights)

            # forward + backward + optimize
            outputs = tnet(x)
            loss = criterion(outputs, y)
            loss.backward(retain_graph=calc_reg, create_graph=False)
            emb_optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:  # print every 2000 mini-batches
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

            # Initialize the regularization loss
            loss_reg = 0

            # Initialize dTheta, the candidate change in the hnet parameters
            dTheta = None

            if calc_reg:
                # Find out the candidate change (dTheta) in trainable parameters (theta) of the hnet
                # This function just computes the change (dTheta), but does not apply it
                dTheta = calc_delta_theta(theta_optimizer,
                                          False,
                                          lr=config["lr"],
                                          detach_dt=True)

                # Calculate the regularization loss using dTheta
                # This implements the second part of equation 2
                loss_reg = calc_fix_target_reg(hnet,
                                               task_id,
                                               targets=targets,
                                               dTheta=dTheta)

                # Multiply the regularization loss with the scaling factor
                loss_reg *= config["beta"]

                # Backpropagate the regularization loss
                loss_reg.backward()

            # Update the hnet params using the current task loss and the regularization loss
            theta_optimizer.step()

    torch.save(hnet, "hnet.pt")
