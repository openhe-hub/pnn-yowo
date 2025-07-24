## PNN Version of Humanoidverse
### Modification (Under /humanoidverse)
* PNN PPO: /agents/pnn_ppo/pnn_ppo.py
* PNN Block & Net: /agents/modules/pnn_modules.py
* PNN Actor/Critic: /agents/modules/pnn_ppo_modules.py
* New Env (Support motion switch): /envs/motion_tracking/motion_trackinh.py
* Training: train_agent_pnn.py
### How it runs
```bash
python humanoidverse/train_agent_pnn.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=debug \
seed=1 \
+device=cuda:0
```
* Modify Task List (Motion *.pkl) at: `config/base.yaml`, `pnn.motions`
### Training Steps 
1. init env
2. for each task
   1. reset motion file in env 
   2. reset env
   3. new task in PPO
   4. new task in actor/critic
   5. freeze previous & new column in PNN
   6. train PPO
