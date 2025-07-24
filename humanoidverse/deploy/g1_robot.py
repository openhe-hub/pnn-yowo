# -*- coding: utf-8 -*-
"""
Low-level Unitree G1 adapter (23-DoF, 250 Hz) for HumanoidVerse / PBHC URCI.
Publishes LowCmd_ on rt/lowcmd and reads LowState_ on rt/lowstate.
"""
from __future__ import annotations
import time, numpy as np
from humanoidverse.deploy.urcirobot import URCIRobot

# DDS plumbing
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
)
# g1_robot.py  ─── head of file
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_  as LowCmd_F,
    unitree_hg_msg_dds__LowState_ as LowState_F)


from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ , LowState_

from unitree_sdk2py.utils.crc import CRC


class Mode:
    PR = 0     # Position / Roll-Pitch series control
    AB = 1     # Ankle A/B parallel control (G1 feet)


# indices of the 23 actuated joints inside the 35-slot LowState array
# ---- 23-DoF G1: legs (0-11), trunk (12-15), arms-no-wrists (16-20, 24-28)
DOF_IDX = [
    0, # left_hip_pitch_joint
    1, # left_hip_roll_joint
    2, # left_hip_yaw_joint
    3, # left_knee_joint
    4, # left_ankle_pitch_joint
    5, # left_ankle_roll_joint  
    6, # right_hip_pitch_joint
    7, # right_hip_roll_joint
    8, # right_hip_yaw_joint
    9, # right_knee_joint
    10, # right_ankle_pitch_joint
    11, # right_ankle_roll_joint
    12, # waist_yaw_joint
    13, # waist_roll_joint
    14, # waist_pitch_joint                      
    15, # left_shoulder_pitch_joint               
    16, # left_shoulder_roll_joint
    17, # left_shoulder_yaw_joint
    18, # left_elbow_joint            
    22, # right_shoulder_pitch_joint
    23, # right_shoulder_roll_joint
    24, # right_shoulder_yaw_joint
    25  # right_elbow_joint         
]   # 23 numbers exactly


class G1Robot(URCIRobot):
    REAL = True                                # safety flag

    def __init__(self, cfg):
        super().__init__(cfg)

        # ---------- DDS initialisation ----------
        ChannelFactoryInitialize(0, cfg.deploy.nic)   

        self.pub = ChannelPublisher("rt/lowcmd",  LowCmd_)  # :contentReference[oaicite:3]{index=3}
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.pub.Init()
        self.sub.Init(self._state_cb, 10)                   # handler, queue size


        # ---------- control parameters ----------
        self.dt = cfg.deploy.ctrl_dt                       # 0.004 s → 250 Hz :contentReference[oaicite:4]{index=4}
        
        self.init_kp_kd()

        self._low_cmd = LowCmd_F() 
        
        self._low_state = None

        self.crc = CRC()
        print("[G1Robot23] Waiting for first LowState packet …")
        print("\n------------\nDT: ", self.dt)
        while self._low_state is None:
            time.sleep(self.dt)
    
    def init_kp_kd(self):
        self.kp = np.zeros(self.num_dofs)
        self.kd = np.zeros(self.num_dofs)
        
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.robot.control.stiffness.keys():
                if dof_name in name:
                    self.kp[i] = self.cfg.robot.control.stiffness[dof_name]
                    self.kd[i] = self.cfg.robot.control.damping[dof_name]
                    found = True
                    print(f"PD gain of joint {name} were defined, setting them to {self.kp[i]} and {self.kd[i]}")
            if not found:
                raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        # for testing purposes only
        self.kp *= 0.1
        self.kd *= 0.1


    @staticmethod
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        '''Calculates torques from position commands
        '''
        print ((target_q - q) * kp + (target_dq - dq) * kd)
        return ((target_q - q) * kp + (target_dq - dq) * kd)
    

    # ---------- URCIRobot required hooks ----------
    def _state_cb(self, msg: LowState_):
        self._low_state = msg

    def _get_state(self):
        s = self._low_state

        self.q  = np.array([s.motor_state[i].q  for i in DOF_IDX],
                           dtype=np.float32)
        self.dq = np.array([s.motor_state[i].dq for i in DOF_IDX],
                           dtype=np.float32)
        
        self.quat  = np.array(s.imu_state.quaternion , dtype=np.float32)
        self.omega = np.array(s.imu_state.gyroscope  , dtype=np.float32)
        self.gvec  = np.array(s.imu_state.accelerometer, dtype=np.float32)



    def _apply_action(self, q_target: np.ndarray):
        """Send PBHC's 23-element position targets plus gains."""
        self._low_cmd.mode_pr = Mode.PR
        self._low_cmd.mode_machine = self._low_state.mode_machine 
        for local_id, lowstate_id in enumerate(DOF_IDX):
            m = self._low_cmd.motor_cmd[lowstate_id]


            q_ref   = float(q_target[local_id])
            dq_ref  = 0.0
            kp      = float(self.kp[local_id])
            kd      = float(self.kd[local_id])
            torque  = float(
                G1Robot.pd_control(q_ref, self.q[local_id], kp,
                                dq_ref, self.dq[local_id], kd)
            )
            m.mode = 1 
            m.q   = q_ref
            m.qd  = 0.0
            m.kp  = kp
            m.kd  = kd
            m.tau = torque * 0.1
        # update CRC each cycle (SDK requirement)
        # print(self._low_cmd)
        print("sending here...")
        # return
        # print(self._low_cmd)
        self._low_cmd.crc = self.crc.Crc(self._low_cmd)
        self.pub.Write(self._low_cmd)
        time.sleep(0.5) # sleep for 10 seconds to avoid sending too many commands
    
    # todo try and implement it 
    def _get_motion_to_save(self):
        # disable motion capture for now
        return 0.0, {}

    # ---------------------------------------------------------------- reset
    def _reset(self):
        # enable motors & go to initial pose (zero-torque)
        self._low_cmd.mode_pr = Mode.PR
        self._apply_action(self.dof_init_pose)
 


