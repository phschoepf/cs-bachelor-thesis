Notes on experiments, ideas, softwares used.

# Doorgym repo

- Running trained policy with rendering: use "--render" flag with enjoy.py
  `python3 enjoy.py --env-name doorenv-v0 --load-name trained_models/sac/doorenv-v0_sac-pull-floatinghook.160.pt --world-path /home/philemon/Desktop/bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --render`
- Finetuning/Restarting training of a checkpointed model: "--pretrained-policy-load" flag with main.py
  `python3 main.py --env-name doorenv-v0 --algo sac --save-name sac-pull-floatinghook-newtraining --world-path /home/philemon/Desktop/bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --pretrained-policy-load trained_models/sac/doorenv-v0_sac-pull-floatinghook.160.pt`

## HNPPO

### ppo-hn

DoorGym rev `0777578dcefe9e7399d5a5195195c03faa3f8d2d`  
Separate hypernetworks for actor and critic, no special training loop (first attempt, useless now)

### ppo-hn2

DoorGym rev `13644f4ba9370622cca5cc5ef30469f085a4be19`  
One hypernetwork for actor and critic weights, training loop as in CLFD.  
Training order: lever -> pull -> round  

DoorGym rev `98dfd666a5e4808d250698a6c83fb4d005cd8a1e`  
Regularization code was fixed between task 0 and 1 (the first task that used regularization). Some issues with task id, there is one more task than there should be.  
Also, we're not sure if the evaluation properly used task IDs or always evaluated with task_id=0


    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn2 --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_floatinghook/ --task-id=0
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn2-task1 --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn2.380.pt --task-id=1
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn2-task1 --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/round_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn2-task1.90.pt--task-id=2

Evaluate:

    python3 enjoy.py --env-name doorenv-v0 --load-name trained_models/hnppo/doorenv-v0_ppo-hn2-task2.160.pt --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/{lever,pull,round}_blue_floatinghook --render --task-id {0,1,2,3}
   
### ppo-hn3

DoorGym rev `5aef28fd1d69a4691bc5919cd3fc9366fd674333`  
Fixed evaluation code to properly use task IDs.  
Training order: pull -> lever -> round

    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn3-task0pull --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --task-id=0
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn3-task1lever --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn3-task0pull.140.pt --task-id=1
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn3-task2round --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/round_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn3-task1lever.120.pt --task-id=2

### ppo-hn4

DoorGym rev `344cf70db6b190b08fb9e011cf8ac6e9edd77d0b`  
Implemented dist weight setting by hnet (new FunctionalDiagGaussian class). Dist_entropy no longer constant (as it should be).  
New save folder structure (also adapted old HNPPO saves to it).

    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn4-task0-pull --world-path /home/philemon/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --task-id=0
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn4-task1-lever --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn4-task0-pull/ppo-hn4-task0-pull.130.pt --task-id=1

Deterministic policy was also tried here in the 2nd task:

    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn4-task1-lever-deterministc --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_floatinghook/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn4-task0-pull/ppo-hn4-task0-pull.130.pt --task-id=1 

(failed, did not learn anything, loss exploded)

### ppo-chn1

DoorGym rev `344cf70db6b190b08fb9e011cf8ac6e9edd77d0b`  
Trying out chunked hypernetworks

    python3 main.py --env-name doorenv-v0 --algo chnppo --save-name ppo-chn1-task0-pull --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --task-id=0
    python3 main.py --env-name doorenv-v0 --algo chnppo --save-name ppo-chn1-task1-lever --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_floatinghook/ --pretrained-policy-load trained_models/chnppo/doorenv-v0_ppo-chn1-task0-pull/ppo-chn1-task0-pull.355.pt --task-id=1

### ppo-hn5 (old bot)

DoorGym rev `344cf70db6b190b08fb9e011cf8ac6e9edd77d0b`  
Testing with `b1-gripper` robot instead of `b1-floatinghook`

    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn5-task0-pull --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_gripper/ --task-id=0

Observation: just hits door and opens it since there is no latch. Trying with more challenging lever and round knobs.

    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn5-task0-lever --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_gripper/ --task-id=0

No opening, only manages to push down lever. 

### ppo-hn5 (new bot)

DoorGym rev `a3866ae62c38ff8d08b3ff5b6f2ea82d8e73cd22`  
Changed robot to start with arm down so it does not collide with doorframe and get kicked off wildly

    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn5-task0-pull --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_gripper/ --task-id=0
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn5-task1-lever --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/lever_blue_gripper/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn5-task0-pull/ppo-hn5-task0-pull.200.pt --task-id=1
    python3 main.py --env-name doorenv-v0 --algo hnppo --save-name ppo-hn5-task2-round --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/round_blue_gripper/ --pretrained-policy-load trained_models/hnppo/doorenv-v0_ppo-hn5-task1-lever/ppo-hn5-task1-lever.180.pt --task-id=2

### ppo-hn6

DoorGym rev `a273261ebad327ebf2978cd89d526815243f93af`  
Critic is no longer part of the hnet, just actor is. Critic gets reset for each training dataset. Fixed a bug where theta optimizer was called 2x per step  
Fixed some parameters in the optimizers

    python3 main.py --env-name doorenv-v0 --algo hnppo --num-processes 12 --save-name ppo-hn6-task0-pull --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world/pull_blue_gripper/ --task-id=0



# Ideas & next steps

- Robomimic repo?

# Initial Presentation

- 2022-03-15
- Presentation: what is CL, video demo of existing doorgym algos, show different handles, (Learning from demostration -> just RL)
