import time
import sys

import joblib

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

FPS = 50
ROS_RATE = 1 / FPS  

G1_NUM_MOTOR = 29

Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms 
]

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

PBHC_JOINT_INDEX = [
  G1JointIndex.LeftHipPitch,
  G1JointIndex.LeftHipRoll,
  G1JointIndex.LeftHipYaw,
  G1JointIndex.LeftKnee,
  G1JointIndex.LeftAnklePitch,
  G1JointIndex.LeftAnkleRoll,
  G1JointIndex.RightHipPitch,
  G1JointIndex.RightHipRoll,
  G1JointIndex.RightHipYaw,
  G1JointIndex.RightKnee,
  G1JointIndex.RightAnklePitch,
  G1JointIndex.RightAnkleRoll,
  G1JointIndex.WaistYaw,
  G1JointIndex.WaistRoll,
  G1JointIndex.WaistPitch,
  G1JointIndex.LeftShoulderPitch,
  G1JointIndex.LeftShoulderRoll,
  G1JointIndex.LeftShoulderYaw,
  G1JointIndex.LeftElbow,
  G1JointIndex.RightShoulderPitch,
  G1JointIndex.RightShoulderRoll,
  G1JointIndex.RightShoulderYaw,
  G1JointIndex.RightElbow,
]


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class PbhcPublisher:
    def __init__(self, onnx_output_file: str, motion_name: str='motion0'):
        self.time_ = 0.0
        self.control_dt_ = 0.02  # [50Hz]
        self.duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()
        # pbhc sim2sim output
        self.onnx_output_file = onnx_output_file
        self.motion_name = motion_name
        self.ReadOnnxOutput()
    
    def ReadOnnxOutput(self):
        self.data = joblib.load(self.onnx_output_file)
        self.data = self.data[self.motion_name]
        self.frame_num = self.data['motion_times'].shape[0]
        self.curr_frame_id = 0
        print(" --> Motion loaded.\n")

    def Init(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        
        self.counter_ +=1
        if (self.counter_ % 500 == 0) :
            self.counter_ = 0
            print(self.low_state.imu_state.rpy)

    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1        # ENABLE SERVO
            self.low_cmd.motor_cmd[i].kp   = Kp[i]    # position gain
            self.low_cmd.motor_cmd[i].kd   = Kd[i]    # velocity gain
            self.low_cmd.motor_cmd[i].dq   = 0.0
            self.low_cmd.motor_cmd[i].tau  = 0.0

        for idx, joint_idx in enumerate(PBHC_JOINT_INDEX):
            self.low_cmd.motor_cmd[joint_idx].q = self.data['action'][self.curr_frame_id][idx]
        
        self.curr_frame_id = (self.curr_frame_id + 1) % self.frame_num 

        if self.curr_frame_id % 50 == 0:
            print(f"Time: {self.time_:.2f} s, Frame ID: {self.curr_frame_id}, Command: \n{self.low_cmd}\n")
        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, 'enp58s0')
        
    onnx_output_file = '/home/airlabresearch/Desktop/humanoid/phbc/example/pretrained_horse_stance_pose/motions/zhewenmotion.pkl'
    publisher = PbhcPublisher(onnx_output_file)
    publisher.Init()
    publisher.Start()

    while True:        
        time.sleep(ROS_RATE)