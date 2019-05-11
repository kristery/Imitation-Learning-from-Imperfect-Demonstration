# Imitation Learning from Imperfect Demonstration
The TRPO part is hugely based on: https://github.com/ikostrikov/pytorch-trpo

## Requirement
 * Python 3.6
 * PyTorch 0.4.1
 * gym
 * mujoco
 * numpy
 * scipy
 
## Execute
The .py files take trajectories and confidence data as inputs (in demonstrations folder) and record accumulated reward at each update in the log folder.
Please follow below commands to run our methods and baselines. Traj-size option is the same as specifying $n_c+n_u$ in the paper and num-epochs specifies the maximum number of update iterations.

 * IC_GAIL
 ```
 python IC_GAIL.py --env Ant-v2 --num-epochs 5000 --traj-size 600 
 ```
 * 2IWIL
 ```
 python 2IWIL.py --env Ant-v2 --num-epochs 5000 --traj-size 600 --weight
 ```
 * GAIL (U+C)
 ```
 python 2IWIL.py --env Ant-v2 --num-epochs 5000 --traj-size 600
 ```
 * GAIL (C)
 ```
 python 2IWIL.py --env Ant-v2 --num-epochs 5000 --traj-size 600 --weight --only --noconf
 ```
 * GAIL (Reweight)
 ```
 python 2IWIL.py --env Ant-v2 --num-epochs 5000 --traj-size 600 --weight --only
 ```

