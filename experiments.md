Notes on experiments, ideas, softwares used.

# Doorgym repo

- Running trained policy with rendering: use "--render" flag with enjoy.py
  `python3 enjoy.py --env-name doorenv-v0 --load-name trained_models/sac/doorenv-v0_sac-pull-floatinghook.160.pt --world-path /home/philemon/Desktop/bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --render`
- Finetuning/Restarting training of a checkpointed model: "--pretrained-policy-load" flag with main.py
  `python3 main.py --env-name doorenv-v0 --algo sac --save-name sac-pull-floatinghook-newtraining --world-path /home/philemon/Desktop/bachelor-thesis/DoorGym/world_generator/world/pull_blue_floatinghook/ --pretrained-policy-load trained_models/sac/doorenv-v0_sac-pull-floatinghook.160.pt`

# Ideas & next steps

- Implement own algorithm, to check out how doorgym uses algos
- Train for longer
- Robomimic repo?

# Initial Presentation

- 2022-03-15
- Presentation: what is CL, video demo of existing doorgym algos, show different handles, (Learning from demostration?)