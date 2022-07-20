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

## lr Optimization

1st round: Without determinism, 4 learning rates `[ 1e-4, 1e-3, 5e-3, 1e-2 ]`  
General observation: lr > 1e-3 is better, should focus on 1e-3...1e-2 range

| Name                   | batch id | machine | result                                                                     |
|------------------------|----------|---------|----------------------------------------------------------------------------|
| lr_opt_pull            | f547a804 | gpu1    | lr 1e-3 and **5e-3** about equally well, both open doors after ~3 envsteps |
| lr_opt_lever           | 21ec6c48 | gpu2    | lr **1e-2** gives highest reward but lots of noise, 5e-3 more stable       |
| lr_opt_round           | 26489180 | gpu1    | lr 5e-3 and 1e-2 about equally well                                        |
| lr_opt_pull_left_fixed | 1d523a55 | gpu3    |                                                                            |

Determinism test: Add set_seed() function, try same hparams twice, with `--cuda-deterministic, --seed 1000` flags, 
see if results are really the same.  
Result: Have to run single-threaded (`--num-processes 1`), otherwise not deterministic. 

| Name                          | batch id | machine | result                                              |
|-------------------------------|----------|---------|-----------------------------------------------------|
| determinism_test              | 9ca4f008 | gpu5    | not deterministic, different world files are picked |
| determinism_test_singlethread | ad89f543 | gpu5    | **deterministic**                                   |

2nd round: deterministic, smaller lr range (determined per-task from 1st round)
