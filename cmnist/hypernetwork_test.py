import sys
from tqdm import trange, tqdm
from clfd.imitation_cl.model.hypernetwork import *
from split_mnist import SplitMNIST


def train_task(task_id, hnet, tnet, config):
    assert task_id in range(0, 5)
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

    criterion = torch.nn.BCEWithLogitsLoss()
    # Start training iterations
    for epoch in trange(5):

        running_loss = 0.0
        for i, (x, y) in enumerate(DataLoader(SplitMNIST("~/Desktop/schoepf-bachelor-thesis/cmnist/data",
                                                         classes=[task_id * 2, task_id * 2 + 1],
                                                         transform=lambda img: np.asarray(img, dtype=np.float32).flatten()),
                                              batch_size=16,
                                              shuffle=True)):

            # bring labels to [0,1] range
            y -= (2*task_id)
            # make a one-hot tensor from the targets
            # 2 classes since we do task-incremental learning (only 2 choices per task and we know that task)
            # domain/class-incremental learning would get the full 10 output classes
            y = F.one_hot(y, num_classes=2).float().to(tnet.device)

            x = x.to(tnet.device)
            ### Train theta and task embedding
            theta_optimizer.zero_grad()
            emb_optimizer.zero_grad()

            # Populate weights of tnet with outputs of hnet
            weights = hnet.forward(task_id)
            tnet.set_weights(weights)

            # forward + backward + optimize
            outputs, logits = tnet(x)
            loss = criterion(logits, y)
            loss.backward(retain_graph=calc_reg, create_graph=False)
            emb_optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                tqdm.write(f'[{epoch}, {i + 1:3d}] loss: {running_loss / 100:.1e}')
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


def evaluate(task_id, hnet, tnet):
    tnet.set_weights(hnet.forward(task_id))

    split_mnist_test = SplitMNIST("~/Desktop/schoepf-bachelor-thesis/cmnist/data",
                                  classes=[task_id * 2, task_id * 2 + 1],
                                  transform=lambda img: np.asarray(img, dtype=np.float32).flatten(),
                                  train=False)
    correct = 0
    for x, y in DataLoader(split_mnist_test):
        x = x.to(hnet.device)
        # bring labels to [0,1] range
        y -= (2 * task_id)
        y = y.to(hnet.device)
        outputs, logits = tnet(x)
        pred = torch.argmax(outputs)
        if pred == y.long():
            correct = correct + 1
    print(f"accuracy: {correct / split_mnist_test.__len__():.3f}")
    return correct / split_mnist_test.__len__()


def init_nets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hnet = HyperNetwork(layers=[2048, 512, 128],
                        te_dim=5,
                        target_shapes=TargetNetwork.weight_shapes(n_in=28 * 28, n_out=2, hidden_layers=[128, 64]),
                        device=device).to(device)

    tnet = TargetNetwork(n_in=28 * 28,
                         n_out=2,
                         hidden_layers=[128, 64],
                         no_weights=True,
                         bn_track_stats=False,
                         activation_fn=torch.nn.ReLU(),
                         out_fn=torch.nn.Sigmoid(),
                         device=device).to(device)

    return hnet, tnet


def eval_all_tasks(max_tid, hnet, tnet):
    accuracies = []
    for tid in range(0, max_tid + 1):
        accuracies.append(evaluate(tid, hnet, tnet))
    print(f"mean accuracy on tasks {list(range(0,max_tid+1))}: {sum(accuracies)/len(accuracies):.3f}")
    return accuracies


if __name__ == "__main__":
    # Examples:
    #       python3 cmnist/hypernetwork_test.py train 0
    #       python3 cmnist/hypernetwork_test.py eval 0
    tid = int(sys.argv[2]) if len(sys.argv) > 1 else 0
    config = {"beta": 5e-3,
              "lr": 1e-4,
              }
    hnet, tnet = init_nets()

    if sys.argv[1] == "train":
        # load net learned from previous task
        hnet = torch.load(f"cmnist/models/hnet{tid-1}.pt")
        train_task(tid, hnet, tnet, config)
        torch.save(hnet, f"cmnist/models/hnet{tid}.pt")
    elif sys.argv[1] == "eval":
        hnet = torch.load(f"cmnist/models/hnet{tid}.pt")
        eval_all_tasks(tid, hnet, tnet)
