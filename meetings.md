# Meeting minutes: Thursday progress meetings

## 2022-01-07

Continual learning for robots
teach robot to do sth, then teach something else -> should do both
scalable to 10-15 tasks in tandem

image 1: red are training data, blue are reproduced data
-> shold be able to learn writing different letters

this work: simulation environment, try out different approaches
* reinforcement learning (not possible w/ real robot)
* kinesthetic teaching: holding robot by hand and "showing" movements
* 3-4 different appraoches in comparaison

examples:
* door opening: different kinds of door handles
* block pushing: blocks with different frictions, surfaces

task 1: research whats out there in terms of simulators
* mb start w/ coppeliaSim
* gazebo simulator
* panda (franka) robot
* will get more links -> prefer open source software

task 2: implement CL algorithms
taks 3: compare performance in simulations

bigger picture: use environment in later work 

will get:
* links for simulators
* papers to read on CL algorithms

meet again in week of 2021-01-17

## 2022-02-03

- how to integrate integrate muojoco in coppelia
- focus on door gym for now?!
- zid VM
- presentation -> 1st or 2nd week of new semester
- algos to compare
- EWC
- memory activated synapse
- synapse intelligence
- our approach: **hypernetworks**
 
 
develop environments
RAL paper? -> learning from demonstration
next: also RL
hypernetwork would generate policy network (for RL)

my work: concentrate on hypernetwork for RL similar to https://arxiv.org/pdf/2009.11997.pdf
change how the hypernetwork works

sayantan will give me sample coppeliasim env
- set up
- get familiar with robots
- RL policy methods
 - TD3
 - SAC
 
look into mujoco licensing -> can we publish with it
get panda robot into door gym

next meeting: prep door envi, demostrate pretrained models
2022-02-17 11:00

## 2022-02-17

initial presentation: 2022-03-15 16:00
repo: put everything in one repo for now, maybe restructure later
also put meeting notes + stuff i currently do in repo (somewhere)

presentation idea: show agent opening a door (with baseline algo like SAC)
or at least attempting to do so

next:
- get SAC runnning -> open a fixed type of door (lever+hook robot)
- presentaiton for initial

Remote GPU on zid -> will get email

## 2022-03-02

use tensorboard to log
also check comments on presentation pdf
will get small RL toy problem from sayantan to check and understand SAC algo

send presentation to Antonio next week (2022-03-11)
10 minutes! 

 TODO in presentation:
 
 ad CL slide: CL is without access to previous training data, more general: multitask learning
 ad previous work: mention types of CL instead of papers: regularization, replay, multi head,...
 ad doorgym slide: 1 slide with fixed points, 1 with optional ideas to expand
  - mention that only 1 task is learned at a time
  - first task gnostic, then task agnostic -> inference task from renderings (CNN)

## 2022-03-17

more tasks -> new door handles?
 - around 10 would be nice for a paper
 - new worlds
 - look into world generation
 report back in next meeting
 
SAC: try to get it working
 - try in simpler environment (halfcheetah)
 - ignore for now

try to get hypernetwork with PPO working!

understand how to use hypernetworks
 - code up a small non-RL hypernetwork
 - regression example
 
chunked hypernetwork -> also try small example
pay attention to size of hypernetwork!

## 2022-03-31

- doors: objective functions!
- clfd: maybe use other (more fundamental) repo


TODO
- eval for task cl as triangular matrix
 -- can compute more metrics with this matrix https://arxiv.org/pdf/1810.13166.pdf
 
task inference CNN also has to be learned continually

ppo: keep default dimensions from doorgym
task embedding and regularization coefficient
notebook for demos

## 2022-04-14

critic dimension 1 is expected! this is the "quality" of the action
actor features are transformed again in distributions.py

use only 1 hypernetwork for the 2 target networks
 - more output dimensions
 - 2 heads
 
todo: 
 - integrate output layer to HNBase (instead of a separate place)
 - make a single HN for actor and critic
 - function to split the outputs to actor and critic network weights


## 2022-04-28

questions:
 - make mujoco faster?
 - network size?
 - SAC? vision?

discussed with jacob as well on RL stuff
new tasks:

 - check embeddings of task ids and target networks are for each task id
 - train on more tasks (up to 5)
 - set up gpu machine on zid
 - evaluate all training steps on all tasks
  -- forward with highest trained task id
  -- backward with respective task id of the task evaluated
  -- also backward with highest task embedding (last trained)
 - then: SAC, TD3

will get email from sayantan with optimization ideas for hypernetworks