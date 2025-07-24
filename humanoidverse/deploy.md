### play trained policy
python humanoidverse/eval_agent.py +device=cuda:0 +env.config.enforce_randomize_motion_start_eval=False +checkpoint=example/pretrained_horse_stance_pose/model_50000.pt

### Sim2Sim (Deploy in Mujoco)

modify `REAL       :bool    = False` in `urci.py`

python humanoidverse/urci.py +checkpoint='example/pretrained_horse_stance_pose/exported/model_50000.onnx' +opt=record +simulator=mujoco

### Sim2Real

modify `REAL       :bool    = True` in `urci.py`

python humanoidverse/urci.py   +simulator=real   +checkpoint=example/pretrained_horse_stance_pose/exported/model_50000.onnx +deploy.BYPASS_ACT=False +deploy.SAVE_REPLAY=False ++deploy.ctrl_dt=0.01
