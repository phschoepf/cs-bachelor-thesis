# Guild Run list

Documentation about the Guild experiments done.

c703i overview:

| Nr. | GPU         | note                                                                     |
|-----|-------------|--------------------------------------------------------------------------|
| 1   | GTX 1080Ti  |                                                                          |
| 2   | GTX 1060    |                                                                          |
| 3   | RTX 2080Ti  |                                                                          |
| 4   | GTX Titan X | no access to shared homes                                                |      
| 5   | RTX 2070    |                                                                          |
| 6   | GTX 1080    |                                                                          |
| 7   | RTX 3090    |                                                                          |
| 8   | RTX 3090    |                                                                          |
| 9   | RTX 3090    | throws CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm(handle)` |

## lr Optimization (hnppo)

1st round: Without determinism, 4 learning rates `[ 1e-4, 1e-3, 5e-3, 1e-2 ]`  
General observation: lr > 1e-3 is better, should focus on 1e-3...1e-2 range

| Name                   | batch id | machine | result                                                                      |
|------------------------|----------|---------|-----------------------------------------------------------------------------|
| lr_opt_pull            | f547a804 | gpu1    | lr 1e-3 and **5e-3** about equally well, both open doors after ~3M envsteps |
| lr_opt_lever           | 21ec6c48 | gpu2    | lr **1e-2** gives highest reward but lots of noise, 5e-3 more stable        |
| lr_opt_round           | 26489180 | gpu1    | lr 5e-3 and 1e-2 about equally well                                         |
| lr_opt_pull_left_fixed | 1d523a55 | gpu3    | same observation as in pull, but already open after 2M steps                |
| lr_opt_lever_left      | 451a3b94 | gpu1    | all very close, but 0.015 seems to work best.                               |

## Determinism 
Setup: Add set_seed() function, try same hparams twice, with `--cuda-deterministic, --seed 1000` flags, 
see if results are really the same.  
Result: Have to run single-threaded (`--num-processes 1`), otherwise not deterministic. 
Fixed problem by passing already chosen worlds into `env.make()`. Can now run deterministic with multithreading!

| Name                          | batch id | machine | result                                              |
|-------------------------------|----------|---------|-----------------------------------------------------|
| determinism_test              | 9ca4f008 | gpu5    | not deterministic, different world files are picked |
| determinism_test_singlethread | ad89f543 | gpu5    | **deterministic**                                   |
| determinism_test              | bd420e38 | gpu1    | fix: choose worlds before fork, **determinisitc**   |

## Bayesian optimization with multiple parameters (ppo)

Params to optimize: `lr, network-size, clip-param`. Optimizing for **vanilla PPO** first.

| Name                       | batch id | machine | result                                                                                                               |
|----------------------------|----------|---------|----------------------------------------------------------------------------------------------------------------------|
| lr_size_clip_opt_ppo       | ce5370cc | gpu3    | best (cluster): lr=0.0027, clip=0.48, net=[64,64]. also good (single point): lr=0.004, clip=0.20, net=[64, 64]       |
| lr_size_clip_opt_ppo_lever | 8452dd9b | gpu1    | best (single point): lr=0.0006, clip=0.49, net=[64,64]. cluster of good values at: lr=0.003, clip=0.33, net=[64, 64] |
| lr_size_clip_opt_ppo_round | 5a75bff6 | gpu3    | best (cluster): lr=1.0e-4, clip=0.597, net=[64,64,64]. Everything above lr=0.005 becomes abysmal or errors out.      |

No opening on lever or round, even after optimization. 

### Long runs with optimized parameters

| Name           | batch id | machine | result                                     |
|----------------|----------|---------|--------------------------------------------|
| ppo_lever_long | 6aa2b881 | gpu6    | no opening, but rather high reward (7696)  |
| ppo_round_long | 4eb3f464 | gpu5    | no opening, reward stagnant after 1M steps |


## Bayesian optimization with multiple parameters (hnppo)

Same optimization as done for vanilla ppo, now for the fresh hypernetwork (task 0, no CL)

| Name                                 | batch id | machine | result                                                                                                                                                                   |
|--------------------------------------|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| lr_size_clip_opt_hnppo_lever         | 45d2abfc | gpu1    | best: lr=0.0074, clip=0.41, net=[64 64 64]. lr definitely has largest impact, good results around 0.007-0.013                                                            |
| lr_size_clip_opt_hnppo_round         | 20028a8e | gpu6    | best: lr=0.019851, clip=0.215917, net=[64 64 64]. All good runs are at the high-limit of the lr search space (0.02) and low-limit of the clipping space (0.2)            |
| lr_size_clip_entropy_opt_hnppo_lever | 8acaee00 | gpu3    | Larger size and depth do not bring advantage, high te-dim, lr=0.005-0.02, clip=0.3-0.4, entropy>1e-4. clip, te-dim and lr seem to be most important. Floatinghook robot. |

General observation: Much fewer runs end in error, maybe due to lower (1e-4) max-grad-norm. Also, the network size does not seem to have much impact.
Result from large 6-parameter search: keep net=[64,64], high lr (>0.01). te-dim=15, entropy 1e-4, clip=0.35. The "66f0572e86ef422c9497612f89d732ff" run looks especially promising.

## Testing with floatinghook robot

The floatinghook was previously able to open lever doors occasionally - try again with the optimized hparams, both hnppo and vanilla ppo.

| Name                      | batch id | machine | result                                                                                                          |
|---------------------------|----------|---------|-----------------------------------------------------------------------------------------------------------------|
| hooktest_lever            | a5858514 | gpu1    | task 0 HNPPO: on the right track to solving task, does find lever and push down on it                           |
| hooktest_lever_ppo_latest | c6499a0f | gpu3    | hparams from latest doorgym paper. **PPO can completely solve the task** with 90% success rate after 12M steps. |
| hnppo_lever_long          | a2ef83d0 | gpu7    | 73% success rate after 340 epochs (paper PPO baseline had 68%                                                   |

## Continual Learning series

After figuring out suitable hparams for the pull and lever task with the floatinghook, we measure its CL performance by training on successive tasks.

| Name                     | batch id | machine | task id | result                                                                                                                                             |
|--------------------------|----------|---------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| hnppo_lever_long         | a2ef83d0 | gpu7    | 0       | 73% success rate after 340 epochs (paper PPO baseline: 68%. Copied from the hparam search.                                                         |
| hnppo_pull_1             | 7eb5622d | gpu2    | 1       | Reusing same hparams as for lever. Run failed after 1st epoch because of gradient explosion. This only happens with the high lr, 5e-3 lr works.    |
| hnppo_pull_1_defaultargs | eee49843 | gpu2    | 1       | 95% success rate after 40 epochs (baseline PPO: 95%), stabilizes between 95%-100% afterwards                                                       |
| hnppo_lever_left_2       | d574e035 | gpu2    | 2       | Failed after 1st epoch, gradient explosion. lr=0.01 instead of 0.02 works, ran out of memory after 220 steps (probably other factors). No opening. |

It seem like the found hparams make multi-task training unstable. Will try again with more default-ish hparams from Doorgym, similar to earlier `ppo-hn10` run.

| Name                       | batch id           | machine | task id | result                                                                            |
|----------------------------|--------------------|---------|---------|-----------------------------------------------------------------------------------|
| series2_hnppo_pull_0       | 8f10319b           | gpu5    | 0       | 100% opening rate after 80 epochs, stopped run at 130 epochs                      |
| series2_hnppo_pull_left_1  | f118768f, 42c6a0d4 | gpu5    | 1       | With pretrained policy at 130: grad explosion. At 120: no opening, learns nothing |

After this: revert hnet width to 10x tnet size (instead of static 1024). Since we use [64,64] as tnet size, this means an overall smaller hnet.  
Also reduced clipping-param back to 0.3 (used originally in hn tests)

| Name                       | batch id | machine | task id | result                                                                                      |
|----------------------------|----------|---------|---------|---------------------------------------------------------------------------------------------|
| series2_hnppo_pull_0       | 93500640 | gpu5    | 0       | task solved after 40 epochs (>>95% opening rate)                                            |
| series2_hnppo_pull_left_1  | 640d6ee1 | gpu5    | 1       | some forward transfer from task0, task fully solved after 20 epochs                         |
| series2_hnppo_lever_2      | c710b1ac | gpu5    | 2       | gets close to solving the task, but does not press the handle down far enough               |
| series2_hnppo_lever_2      | 8956a6e8 | gpu5    | 2       | trying smaller beta=2e-3 to give task more room to learn: 93% opening rate after 360 epochs |
| series2_hnppo_lever_left_3 | b89db3e6 | gpu5    | 3       |                                                                                             |

After CL failures in previous runs: trying beta=1e-3 on all runs, since no forgetting occurred at all so far

| Name                       | batch id | machine | task id | result                                                                         |
|----------------------------|----------|---------|---------|--------------------------------------------------------------------------------|
| series2_hnppo_pull_left_1  | d788a5af | gpu3    | 1       | Result: about same performance as with beta=5e-3                               |
| series2_hnppo_lever_2      | ed78fd87 | gpu3    | 2       | much faster than run with higher beta, 95% after 120 epochs                    |
| series2_hnppo_lever_left_3 | 14626170 | gpu3    | 3       | 91% opening rate after 700 epochs, really slow progress compared to prev tasks |
| series2_hnppo_lever_push_4 | cb74d12b | gpu3    | 4       | 38% opening rate after 80 epochs, drops back to 0 after that                   |

series3: deterministic and even power beta. cuda-deterministic=True seed=1 beta=5e-4

| Name                       | batch id | machine | task id | result                                                                   |
|----------------------------|----------|---------|---------|--------------------------------------------------------------------------|
| series3_hnppo_pull_0       | d1afae8a | gpu5    | 0       | 100% success after 80 epochs                                             |
| series3_hnppo_pull_left_1  | c6c99906 | gpu5    | 1       | 71% success at zero-shot, 100% success after 20 episodes                 |
| series3_hnppo_lever_2      | 101d2761 | gpu5    | 2       | no opening - strange, worked very well in series2                        |
| series3_hnppo_lever_2      | de6b9553 | gpu5    | 2       | trying again with later checkpoint of pull_left - 86% after 365 episodes |
| series3_hnppo_lever_left_3 | 8a902800 | gpu5    | 3       | no progress after 365 epochs                                             |

series4: back to beta=1e-3, worked better. cuda-deterministic=True seed=1

| Name                            | batch id | machine | task id | result                 |
|---------------------------------|----------|---------|---------|------------------------|
| series3_hnppo_pull_0            | d1afae8a | gpu5    | 0       | reused from series3    |
| series4_hnppo_pull_left_1       | 3a46bb8d | gpu1    | 1       | 100% after 60 episodes |
| series4_hnppo_lever_2           | 7f28c4a1 | gpu1    | 2       | 97% after 180 episodes |
| series4_hnppo_lever_left_3      | b4c0b302 | gpu1    | 3       | 86% after 340 episodes |
| series4_hnppo_lever_push_4      | 1ad1b12e | gpu1    | 4       |                        |
| series4_hnppo_lever_push_left_5 |          |         | 5       |                        |

series5: same as series4, but without freshcritic. Critic is in hnet for these runs. 

| Name                            | batch id | machine | task id | result                 |
|---------------------------------|----------|---------|---------|------------------------|
| series5_hnppo_pull_0            | 89acf458 | gpu5    | 0       | 98% after 60 episodes  |
| series5_hnppo_pull_left_1       | f84957b5 | gpu5    | 1       | 100% after 80 episodes |
| series5_hnppo_lever_2           | 7a70a048 | gpu5    | 2       | 89% after 340 episodes |
| series5_hnppo_lever_left_3      | 4ff63c96 | gpu5    | 3       |                        |
| series5_hnppo_lever_push_4      |          |         | 4       |                        |
| series5_hnppo_lever_push_left_5 |          |         | 5       |                        |
