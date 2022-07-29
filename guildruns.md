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