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

series3: deterministic and even lower beta. cuda-deterministic=True seed=1 beta=5e-4

| Name                       | batch id | machine | task id | result                                                                   |
|----------------------------|----------|---------|---------|--------------------------------------------------------------------------|
| series3_hnppo_pull_0       | d1afae8a | gpu5    | 0       | 100% success after 80 epochs                                             |
| series3_hnppo_pull_left_1  | c6c99906 | gpu5    | 1       | 71% success at zero-shot, 100% success after 20 episodes                 |
| series3_hnppo_lever_2      | 101d2761 | gpu5    | 2       | no opening - strange, worked very well in series2                        |
| series3_hnppo_lever_2      | de6b9553 | gpu5    | 2       | trying again with later checkpoint of pull_left - 86% after 365 episodes |
| series3_hnppo_lever_left_3 | 8a902800 | gpu5    | 3       | no progress after 365 epochs                                             |

series4: HNPPO with freshcritic
back to beta=1e-3, worked better. cuda-deterministic=True

| Name                            | batch id | machine | task id | seed  | result                                                                       |
|---------------------------------|----------|---------|---------|-------|------------------------------------------------------------------------------|
| series3_hnppo_pull_0            | d1afae8a | gpu5    | 0       | 1     | reused from series3                                                          |
| series4_hnppo_pull_left_1       | 3a46bb8d | gpu1    | 1       | 1     | 100% after 60 episodes                                                       |
| series4_hnppo_lever_2           | 7f28c4a1 | gpu1    | 2       | 1     | 97% after 180 episodes                                                       |
| series4_hnppo_lever_left_3      | b4c0b302 | gpu1    | 3       | 1     | 86% after 340 episodes                                                       |
| series4_hnppo_lever_push_4      | 1ad1b12e | gpu1    | 4       | 1     | 24% after 365 episodes, just started to solve the task when experiment ended |
| series4_hnppo_lever_push_4      | ee0e6b67 | gpu1    | 4       | 1     | rerun with longer duration: 84% after 660 episodes                           |
| series4_hnppo_lever_push_left_5 | 4c5a8316 | gpu1    | 5       | 1     | highest reward of all, but no opening after 720 episodes                     |                                                          
| ------------------------------- | -------- | ------- | ------- | ----- | ------------------------------------------------------------------------     |
| series4_hnppo_pull_0            | f188b15a | gpu1    | 0       | 31415 | 100% after 60 epis                                                           |
| series4_hnppo_pull_left_1       | e71ded2f | gpu1    | 1       | 31415 | 100% after 120 epis                                                          |
| series4_hnppo_lever_2           | a5f6790c | gpu1    | 2       | 31415 | start from early checkpoint 135, 14% after 360 epis                          |
| series4_hnppo_lever_2           | 8780f15c | gpu2    | 2       | 31415 | start from later checkpoint 230: no opening                                  |
| series4_hnppo_lever_left_3      | a4c50732 | gpu5    | 3       | 31415 | error with NaN after 680 epis, no opening                                    |
| series4_hnppo_lever_push_4      | 2267b32d | gpu5    | 4       | 31415 | 2% after 260 epis, then decline back to 0                                    |
| series4_hnppo_lever_push_left_5 | 3915ca6b | gpu5    | 5       | 31415 | 5% after 280 epis, then decline back to 0                                    |
| ------------------------------- | -------- | ------- | ------- | ----- | ------------------------------------------------------------------------     |
| series4_hnppo_pull_0            | b671948b | gpu1    | 0       | 27182 | 100% after 80 epis                                                           |
| series4_hnppo_pull_left_1       | 4de2fa91 | gpu1    | 1       | 27182 | 100% after 80 epis, erratic training graph. Stabilized after 300             |
| series4_hnppo_lever_2           | 08792b9e | gpu1    | 2       | 27182 | 92% after 260 epis                                                           |
| series4_hnppo_lever_left_3      | 2ab59e97 | gpu1    | 3       | 27182 | 22% after 240 epis, then decline                                             |
| series4_hnppo_lever_push_4      | 292fca9f | gpu1    | 4       | 27182 | error with NaN after 180 epis, no opening                                    |
| series4_hnppo_lever_push_left_5 | 24b3d6d6 | gpu1    | 5       | 27182 | error with NaN after 120 epis, no opening                                    |

series5: same as series4, but without freshcritic. Critic is in hnet for these runs. 

| Name                            | batch id | machine | task id | seed   | result                                                           |
|---------------------------------|----------|---------|---------|--------|------------------------------------------------------------------|
| series5_hnppo_pull_0            | 89acf458 | gpu5    | 0       | 1      | 98% after 60 episodes                                            |
| series5_hnppo_pull_left_1       | f84957b5 | gpu5    | 1       | 1      | 100% after 80 episodes                                           |
| series5_hnppo_lever_2           | 7a70a048 | gpu5    | 2       | 1      | 89% after 340 episodes                                           |
| series5_hnppo_lever_left_3      | 4ff63c96 | gpu5    | 3       | 1      | 88% after 340 episodes, very similar learning profile as lever_2 |
| series5_hnppo_lever_push_4      | 21d66fe7 | gpu5    | 4       | 1      | 87% affter 400 epochs                                            |
| series5_hnppo_lever_push_left_5 | 3f088edd | gpu5    | 5       | 1      | no opening after 720 episodes                                    |
| --------------------------      | -------  | ------- | ------  | ------ | ------------------                                               |
| series5_hnppo_pull_0            | 0f35ca58 | gpu2    | 0       | 31415  | 100% after 40 episodes                                           |
| series5_hnppo_pull_left_1       | f966d792 | gpu2    | 1       | 31415  | 100% after 100 epis                                              |
| series5_hnppo_lever_2           | a52c29fc | gpu2    | 2       | 31415  | 97% after 220 epis                                               |
| series5_hnppo_lever_left_3      | 623cf220 | gpu5    | 3       | 31415  | 60% after 680 epis. used cp@360 for fairness in next run         |
| series5_hnppo_lever_push_4      | a37e8498 | gpu5    | 4       | 31415  | no opening after 720 epis                                        |
| series5_hnppo_lever_push_left_5 | 6100665a | gpu5    | 5       | 31415  | error with NaN after 80 epis, 46% opening                        |
| --------------------------      | -------  | ------- | ------  | ------ | ------------------                                               |
| series5_hnppo_pull_0            | 329e2779 | gpu3    | 0       | 27182  | 100% after 60 episodes                                           |
| series5_hnppo_pull_left_1       | 01730091 | gpu3    | 1       | 27182  | 100% after 80 episodes                                           |
| series5_hnppo_lever_2           | 9b399ea3 | gpu3    | 2       | 27182  | 96% after 160 epis                                               |
| series5_hnppo_lever_left_3      | 7e66b022 | gpu3    | 3       | 27182  | error with NaN after 320 epis, no opening                        |
| series5_hnppo_lever_push_4      | b1685c51 | gpu3    | 4       | 27182  | 91% after 680 epis                                               |
| series5_hnppo_lever_push_left_5 | 34f1ba7c | gpu1    | 5       | 27182  | 50% after 80 epis, then got stuck and did not recover            |

series6: baseline for series5 (each task has a fresh hypernetwork and task_id=0)

| Name                            | batch id | machine | task id | seed   | result                                            |
|---------------------------------|----------|---------|---------|--------|---------------------------------------------------|
| series6_hnppo_pull_0            | 89acf458 | gpu5    | 0       | 1      | reused from series5                               |
| series6_hnppo_pull_left_1       | 8470a8dc | gpu3    | 0       | 1      | 100% after 60 episodes                            |
| series6_hnppo_lever_2           | ce5e9c44 | gpu5    | 0       | 1      | no opening after 365 episodes, but could be close |
| series6_hnppo_lever_left_3      | d7ac3119 | gpu4    | 0       | 1      | 98% after 480 episodes (slower than CL version)   |
| series6_hnppo_lever_push_4      | 7fb8df14 | gpu5    | 0       | 1      | 89% after 720 episodes, learned right at end      |
| series6_hnppo_lever_push_left_5 | 2ed50347 | gpu3    | 0       | 1      | 98% after 720 episodes, very linear progress      |
| --------------------------      | -------  | ------- | ------  | ------ | ------------------                                |
| series6_hnppo_pull_0            | 0f35ca58 | gpu2    | 0       | 31415  | reused from series5                               |
| series6_hnppo_pull_left_1       | 764114a3 | gpu4    | 0       | 31415  | 100% after 100 epis                               |
| series6_hnppo_lever_2           | a308f1ee | gpu4    | 0       | 31415  | no opening after 360 epis                         |
| series6_hnppo_lever_left_3      | 1e55ce3b | gpu2    | 0       | 31415  | no opening after 580 epis                         |
| series6_hnppo_lever_push_4      | 0a17b6ff | gpu4    | 0       | 31415  | 93% after 700 epis                                |
| series6_hnppo_lever_push_left_5 | 7594b0ce | gpu2    | 0       | 31415  | 76% after 720 epis                                |
| --------------------------      | -------  | ------- | ------  | ------ | ------------------                                |
| series6_hnppo_pull_0            | 329e2779 | gpu3    | 0       | 27182  | reused from series5                               |
| series6_hnppo_pull_left_1       | b94e613a | gpu1    | 0       | 27182  | 100% after 60 epis                                |
| series6_hnppo_lever_2           | f2aa0aa8 | gpu1    | 0       | 27182  | 83% after 360 epis                                |
| series6_hnppo_lever_left_3      | a5e2ec36 | gpu6    | 0       | 27182  | no opening after 640 epis                         |
| series6_hnppo_lever_push_4      | 34e6fe75 | gpu1    | 0       | 27182  | no opening after 720 epis, peak: 4%               |
| series6_hnppo_lever_push_left_5 | e32a961b | gpu6    | 0       | 27182  | 33% after 720 epis                                |

series7: ppo-vanilla baseline, hparams from doorgym paper. deterministic, seed=[1, 31415, 27182]

| Name                          | batch id | machine | seed   | result                                 |
|-------------------------------|----------|---------|--------|----------------------------------------|
| series7_ppo_pull_0            | 01047ec5 | gpu4    | 1      | 100% after 60 epis                     |
| series7_ppo_pull_left_1       | 0f84eb84 | gpu4    | 1      | 100% after 40 epis                     |
| series7_ppo_lever_2           | 2aabae2c | gpu4    | 1      | 92% after 360 epis                     |
| series7_ppo_lever_left_3      | afb39543 | gpu4    | 1      | 99% after 280 epis, then decline to 94 |
| series7_ppo_lever_push_4      | 97ae0d2c | gpu4    | 1      | 93% after 160 epis                     |
| series7_ppo_lever_push_left_5 | f811d8b5 | gpu4    | 1      | 94% after 280 epis                     |
| --------------------------    | -------  | ------- | ------ | ------                                 |
| series7_ppo_pull_0            | 896b2d3b | gpu5    | 31415  | 100% after 100 epis                    |
| series7_ppo_pull_left_1       | 862704fd | gpu5    | 31415  | 100% after 100 epis                    |
| series7_ppo_lever_2           | 148440a8 | gpu5    | 31415  | no opening                             |
| series7_ppo_lever_left_3      | ca7dde3e | gpu5    | 31415  | 5% after 240 epis                      |
| series7_ppo_lever_push_4      | aaabfab9 | gpu5    | 31415  | 100% after 680 epis                    |
| series7_ppo_lever_push_left_5 | c4710a26 | gpu5    | 31415  | 32% after 640 epis                     |
| --------------------------    | -------  | ------- | ------ | ------                                 |
| series7_ppo_pull_0            | 9a1382a0 | gpu3    | 27182  | 100% after 160 epis                    |
| series7_ppo_pull_left_1       | a4b4629f | gpu3    | 27182  | 100% after 60 epis                     |
| series7_ppo_lever_2           | a11ea04e | gpu3    | 27182  | no opening                             |
| series7_ppo_lever_left_3      | 68ed18ca | gpu3    | 27182  | no opening                             |
| series7_ppo_lever_push_4      | a409b86d | gpu3    | 27182  | 92% after 180 epis                     |
| series7_ppo_lever_push_left_5 | 57ba3e52 | gpu4    | 27182  | 94% after 200 epis                     |

evals for cl-timeline plot running on gpu4 screen

series8: ppo-finetuning, it's like series7 but we re-use the agents for each new task

| Name                          | batch id | machine | seed   | result                                 |
|-------------------------------|----------|---------|--------|----------------------------------------|
| series7_ppo_pull_0            | 01047ec5 | gpu4    | 1      | 100% after 60 epis                     |
| series8_ppo_pull_left_1       | 87f5ee7c | gpu4    | 1      | 70% zeroshot, 100% after 40 epis       |
| series8_ppo_lever_2           | c6c2d30a | gpu4    | 1      | no opening @365                        |
| series8_ppo_lever_left_3      | dfeb0fc0 | gpu4    | 1      | no opening @365                        |
| series8_ppo_lever_push_4      | 29daa9e4 | gpu4    | 1      | 95% after 420 epis                     |
| series8_ppo_lever_push_left_5 | 1a48e400 | gpu4    | 1      | 93% after 440 epis                     |
| --------------------------    | -------  | ------- | ------ | ------                                 |
| series7_ppo_pull_0            | 896b2d3b | gpu5    | 31415  | 100% after 100 epis                    |
| series8_ppo_pull_left_1       | 9db980b1 | gpu2    | 31415  | 100% after 100 epis                    |
| series8_ppo_lever_2           | 885557cc | gpu2    | 31415  | no opening @365                        |
| series8_ppo_lever_left_3      | 1057a855 | gpu2    | 31415  | no opening @365                        |
| series8_ppo_lever_push_4      | e6f583f2 | gpu2    | 31415  | 45% after 340 epis, then back to 0     |
| series8_ppo_lever_push_left_5 | bc4aafca | gpu3    | 31415  | 25% after 340 epis, erratic afterwards |
| --------------------------    | -------  | ------  | ------ | ------                                 |
| series7_ppo_pull_0            | 9a1382a0 | gpu3    | 27182  | 100% after 160 epis                    |
| series8_ppo_pull_left_1       | 0868504a | gpu3    | 27182  | 100% after 20 epis                     |
| series8_ppo_lever_2           | 92b23643 | gpu3    | 27182  | no opening                             |
| series8_ppo_lever_left_3      | a70ffcb6 | gpu3    | 27182  | no opening @365                        |
| series8_ppo_lever_push_4      | d6414480 | gpu3    | 27182  | no opening @720                        |
| series8_ppo_lever_push_left_5 | 352bd577 | gpu3    | 27182  | 15% after 40 epis, then back to 0      |

series9: Regularizer ablation study. HNPPO with freshcritic (like series4), but beta=0

| Name                            | batch id | machine | task id | seed  | result                                      |
|---------------------------------|----------|---------|---------|-------|---------------------------------------------|
| series3_hnppo_pull_0            | d1afae8a | gpu5    | 0       | 1     | reused from series3                         |
| series9_hnppo_pull_left_1       | f9d1f4f7 | gpu4    | 1       | 1     | 100% after 60 epis                          |
| series9_hnppo_lever_2           | 8319c10c | gpu4    | 2       | 1     | 97% after 260 epis                          |
| series9_hnppo_lever_left_3      | d6a623c5 | gpu4    | 3       | 1     | 98% after 260 epis                          |
| series9_hnppo_lever_push_4      | b2db47ab | gpu4    | 4       | 1     | 95% after 300 epis                          |
| series9_hnppo_lever_push_left_5 | f796fd60 | gpu4    | 5       | 1     | no opening @720                             |                                                          
| ------------------------------- | -------- | ------- | ------- | ----- | -------                                     |
| series4_hnppo_pull_0            | f188b15a | gpu1    | 0       | 31415 | reused from series4                         |
| series9_hnppo_pull_left_1       | b3bf8c3d | gpu1    | 1       | 31415 | 100% after 160 epis                         |
| series9_hnppo_lever_2           | 78f6084c | gpu1    | 2       | 31415 | 48% after 360 epis                          |
| series9_hnppo_lever_left_3      | a3ae4845 | gpu1    | 3       | 31415 | 93% after 360 epis                          |
| series9_hnppo_lever_push_4      | b1f9fd0c | gpu1    | 4       | 31415 | 100% after 400 epis                         |
| series9_hnppo_lever_push_left_5 | 72eccbd6 | gpu1    | 5       | 31415 | no opening @720 epis                        |
| ------------------------------- | -------- | ------- | ------- | ----- | ------                                      |
| series4_hnppo_pull_0            | b671948b | gpu1    | 0       | 27182 | reused from series4                         |
| series9_hnppo_pull_left_1       | b6438d2d | gpu5    | 1       | 27182 | 100% after 40 epis, but erratic graph after |
| series9_hnppo_lever_2           | 8c541921 | gpu5    | 2       | 27182 | 2% after 60 epis, then back to 0            |
| series9_hnppo_lever_left_3      | e64337b6 | gpu5    | 3       | 27182 | no opening @360                             |
| series9_hnppo_lever_push_4      | 811055e9 | gpu5    | 4       | 27182 | 60% after 200 epis, then erratic            |
| series9_hnppo_lever_push_left_5 | 501fadc4 | gpu5    | 5       | 27182 | 17%@340, then back to 0                     |