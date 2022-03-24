#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Compute the error between the estimated poses and the ground true"""

# from distutils.command.build_scripts import first_line_re
from os import path
import csv
# from turtle import pos
import numpy as np
import math

es_poses_path = "/home/lj/Documents/vins-fusion-gpu-no-ros/vins_estimator/build/VIO.txt"
gt_poses_path = "/home/lj/data/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"

interpolation_flag = 0
TimeStampEstimated = 0
TimeStampGroundTruth = 1
es_dt = 0.05
gt_dt = 0.005

# convert quaterion to rotation matrix
def quat2rot(w,x,y,z):
    q0 = w
    q1 = x
    q2 = y
    q3 = z

    r11 = 2*(q0**2+q1**2)-1
    r12 = 2*(q1*q2-q0*q3)
    r13 = 2*(q1*q3+q0*q2)

    r21 = 2*(q1*q2+q0*q3)
    r22 = 2*(q0**2+q2**2)-1
    r23 = 2*(q2*q3-q0*q1)

    r31 = 2*(q1*q3-q0*q2)
    r32 = 2*(q2*q3+q0*q1)
    r33 = 2*(q0**2+q3**2)-1

    rotation = np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
    return rotation

# combine translation and rotation into a 4×4 transformation matrix
def tr2Transform(translation,rotation):
    transformation = np.zeros([4,4],float)
    transformation[0:3,0:3] = rotation;
    transformation[0:3,3] = translation.transpose();
    transformation[3,3] = 1
    return transformation

# computer square sum of the translation from a transformation matrix
def transSquare(M):
    x = M[0,0]
    y = M[1,0]
    z = M[2,0]
    # print("x,y,z = ",x,y,z)
    return x**2+y**2+z**2

# save pose information
class Pose():
    timestamp = 0.0
    p_x = 0.0
    p_y = 0.0
    p_z = 0.0

    q_w = 0.0
    q_x = 0.0
    q_y = 0.0
    q_z = 0.0

    # v_x = 0.0
    # v_y = 0.0
    # v_z = 0.0
    def __init__(self):
        return

    def init1(pose_str, timestampsource):
        pose = Pose()

        coef_TimeStamp = 1.0
        if timestampsource==TimeStampEstimated:
            coef_TimeStamp = 1.0
        elif timestampsource==TimeStampGroundTruth:
            coef_TimeStamp = 1e-9
        else:
            print("Wrong timestampsource.")
            return

        arr = []
        line = pose_str.strip()
        raw = line.split()
        for elem in raw:
            arr.append(float(elem))

        pose.timestamp = arr[0] * coef_TimeStamp
        pose.p_x = arr[1]
        pose.p_y = arr[2]
        pose.p_z = arr[3]
        
        pose.q_w = arr[4]
        pose.q_x = arr[5]
        pose.q_y = arr[6]
        pose.q_z = arr[7]

        return pose

    def init2(timestamp_, p_x_, p_y_, p_z_, q_w_, q_x_, q_y_, q_z_):
        pose = Pose()
        pose.timestamp = timestamp_
        pose.p_x = p_x_
        pose.p_y = p_y_
        pose.p_z = p_z_
        
        pose.q_w = q_w_
        pose.q_x = q_x_
        pose.q_y = q_y_
        pose.q_z = q_z_
        return pose
    
    def merge(timestamp_,pose1,pose2,interpolation_flag_):
        # print("*****************************************************************************")
        # print("【midtimestamp:",timestamp_,"】")

        # pose1.printValue();
        # pose2.printValue();

        time1 = pose1.timestamp
        time2 = pose2.timestamp
        mid_time = (time1 + time2) / 2.0
        if interpolation_flag_ == 0:
            if timestamp_ <= mid_time:
                return pose1
            else:
                return pose2
        elif interpolation_flag_ == 1:
            # 位姿插值
            # print("Time:",time1,time1)
            h = (timestamp_-time1) / (time2-time1)
            # print("h = ",h)
            p_x_ = (1.0-h)*pose1.p_x+h*pose2.p_x
            p_y_ = (1.0-h)*pose1.p_y+h*pose2.p_y
            p_z_ = (1.0-h)*pose1.p_z+h*pose2.p_z
            q_w_ = (1.0-h)*pose1.q_w+h*pose2.q_w
            q_x_ = (1.0-h)*pose1.q_x+h*pose2.q_x
            q_y_ = (1.0-h)*pose1.q_y+h*pose2.q_y
            q_z_ = (1.0-h)*pose1.q_z+h*pose2.q_z
            norm = (q_w_**2+q_x_**2+q_y_**2+q_z_**2)**0.5
            q_w_ = q_w_ / norm
            q_x_ = q_x_ / norm
            q_y_ = q_y_ / norm
            q_z_ = q_z_ / norm
            pose_interpolated = Pose.init2(timestamp_,p_x_,p_y_,p_z_,q_w_,q_x_,q_y_,q_z_)
            # pose_interpolated.printValue()
            return pose_interpolated

    def transformation(self):
        translation = np.array([self.p_x, self.p_y, self.p_z])
        rotation = quat2rot(self.q_w, self.q_x, self.q_y, self.q_z)
        return tr2Transform(translation, rotation)

    def printValue(self):
        print("\nTimestamp",self.timestamp)
        print("Translation:[x,y,z] = [",self.p_x,self.p_y,self.p_z,"]")
        print("Quaterion:[w,x,y,z] = [",self.q_w,self.q_x,self.q_y,self.q_z,"]\n")

class TimeRegistration():
    es_poses = []
    gt_poses = []
    es_poses_selected = []
    gt_poses_selected = []

    def __init__(self):
        
        # 读取估计位姿
        with open(es_poses_path) as f:
            count = 0
            contents = f.readlines()

            while (len(contents[count].split(" "))!=12):
                count = count+1
            contents = contents[count:]
            for line in contents:
                self.es_poses.append(Pose.init1(line,TimeStampEstimated))
            print("Num of es_poses:",len(self.es_poses))
            print(self.es_poses[0].timestamp)

        # 读取ground truth
        with open(gt_poses_path) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=' ', quotechar='|')
            count2 = 0
            for row in csv_reader:
                if(count2==0):
                    count2 = count2+1
                    continue
                self.gt_poses.append(Pose.init1((" ".join(row)).replace(","," "),TimeStampGroundTruth))
            print("Num of gt_poses:",len(self.gt_poses))
            print(self.gt_poses[0].timestamp)
    
    def register(self):
        es_first_flag = 0
        es_last_flag = len(self.es_poses)-1
        gt_flag = 0

        # 时间配准
        # while self.es_poses[es_first_flag].timestamp < self.gt_poses[0].timestamp - (gt_dt/2)*(1-interpolation_flag):
        while self.es_poses[es_first_flag].timestamp < self.gt_poses[0].timestamp:
            es_first_flag = es_first_flag + 1
        while self.es_poses[es_last_flag].timestamp > self.gt_poses[-1].timestamp:
            es_last_flag = es_last_flag - 1
        self.es_poses_selected = self.es_poses[es_first_flag:]
        for es_pose in self.es_poses_selected:
            es_timestamp = es_pose.timestamp
            while es_timestamp < self.gt_poses[gt_flag].timestamp or es_timestamp > self.gt_poses[gt_flag+1].timestamp:
                gt_flag = gt_flag + 1
            self.gt_poses_selected.append(Pose.merge(es_timestamp,self.gt_poses[gt_flag],self.gt_poses[gt_flag+1],interpolation_flag))
    
    def computeError(self):
        if(len(self.es_poses_selected)!=len(self.gt_poses_selected)):
            print("【Error】The number of selected estimated and ground truth poses is not equal.")
            return
        
        RMSE = 0.0
        num_pose = len(self.es_poses_selected)
        print("Num of poses computed:",num_pose)
        for i in range(num_pose-1):
            # Ei = (Q1^-1 · Q2)^-1 · (P1^-1 · P2)
            P1 = self.es_poses_selected[i].transformation()
            P2 = self.es_poses_selected[i+1].transformation()
            Q1 = self.gt_poses_selected[i].transformation()
            Q2 = self.gt_poses_selected[i+1].transformation()

            Ei = np.linalg.inv(np.linalg.inv(Q1) * Q2) * (np.linalg.inv(P1) * P2)
            RMSE += transSquare(Ei)
        RMSE = math.sqrt(RMSE/float(num_pose))
        print("RMSE = ",RMSE)

        
        # RMSE = sqrt{ SUM[ (trans(Ei))^2 ] / m }
    

def main(args=None):
    time_registration = TimeRegistration()
    time_registration.register()
    time_registration.computeError()

    # print(Pose.init1("1403715323.012144 -0.829233 -3.422542 0.505557 0.549673 0.131937 -0.821679 0.072769 -0.468353 0.379462 0.040375 ",TimeStampEstimated).transformation())

    # 完成时间同步
    # 计算相对位姿误差


if __name__ == '__main__':
    main()