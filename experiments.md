Notes on experiments, ideas, softwares used.

# Doorgym repo

- Running trained policy with rendering: use "--render" flag with enjoy.py
  `python3 enjoy.py --env-name doorenv-v0 --load-name trained_models/sac/doorenv-v0_sac-pull-floatinghook.160.pt --world-path /home/philemon/Desktop/bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --render`
- Finetuning/Restarting training of a checkpointed model: "--pretrained-policy-load" flag with main.py
  `python3 main.py --env-name doorenv-v0 --algo sac --save-name sac-pull-floatinghook-newtraining --world-path /home/philemon/Desktop/bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --pretrained-policy-load trained_models/sac/doorenv-v0_sac-pull-floatinghook.160.pt`

## HNPPO

### ppo-hn

Separate hypernetworks for actor and critic, no special training loop (first attempt, useless now)

### ppo-hn2

One hypernetwork for actor and critic weights, training loop as in CLFD.  
Training order: lever -> pull -> round  
Regularization code was fixed between task 0 and 1 (the first task that used regularization). Some issues with task id, there is one more task than there should be.  
Also, we're not sure if the evaluation properly used task IDs or always evaluated with task_id=0


    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn2 --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_floatinghook/ --task-id=0
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn2-task1 --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn2.380.pt --task-id=1
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn2-task1 --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/round_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn2-task1.90.pt--task-id=2

Evaluate:

   python3 enjoy.py --env-name doorenv-v0 --load-name trained_models/hnppo/doorenv-v0_ppo-hn2-task2.160.pt --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/{lever,pull,round}_blue_floatinghook --render --task-id {0,1,2,3}
   
### ppo-hn3

Fixed evaluation code to properly use task IDs.  
Training order: pull -> lever -> round

    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn3-task0pull --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --task-id=0
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn3-task1lever --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn3-task0pull.140.pt --task-id=1
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn3-task2round --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/round_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn3-task1lever.120.pt --task-id=2

# Ideas & next steps

- Robomimic repo?

# Initial Presentation

- 2022-03-15
- Presentation: what is CL, video demo of existing doorgym algos, show different handles, (Learning from demostration -> just RL)
