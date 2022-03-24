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

def quat2rot(w,x,y,z):
    cw = math.cos(w)
    sw = math.sin(w)
    r11 = cw + x**2 * (1-cw)
    r12 = x * y * (1-cw)
    r13 = y * sw
    r21 = r12
    r22 = cw + y**2 * (1-cw)
    r23 = -x * sw
    r31 = -r13
    r32 = -r23
    r33 = cw
    rotation = np.array([r11,r12,r13],[r21,r22,r23],[r31,r32,r33])
    return rotation


def tr2Transform(translation,rotation):
    transformation = np.zeros(4,4)
    transformation[0:2,0:2] = rotation;
    transformation[0:2,3] = translation.transpose;
    transformation[3,3] = 1
    return transformation


class pose():
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

    def __init__(self, pose_str, timestampsource):
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

        self.timestamp = arr[0] * coef_TimeStamp
        self.p_x = arr[1]
        self.p_y = arr[2]
        self.p_z = arr[3]
        
        self.q_w = arr[4]
        self.q_x = arr[5]
        self.q_y = arr[6]
        self.q_z = arr[7]

    def __init__(self, timestamp_, p_x_, p_y_, p_z_, q_w_, q_x_, q_y_, q_z_):
        self.timestamp = timestamp_
        self.p_x = p_x_
        self.p_y = p_y_
        self.p_z = p_z_
        
        self.q_w = q_w_
        self.q_x = q_x_
        self.q_y = q_y_
        self.q_z = q_z_
    
    def merge(timestamp_,pose1,pose2,interpolation_flag_):
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
            return pose1

    def transformation(self):
        translation = np.array([self.p_x, self.p_y, self.p_z])
        rotation = quat2rot(self.q_w, self.q_x, self.q_y, self.q_z)
        return tr2Transform(translation, rotation)

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
                self.es_poses.append(pose(line,TimeStampEstimated))
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
                self.gt_poses.append(pose((" ".join(row)).replace(","," "),TimeStampGroundTruth))
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
            while es_timestamp < self.gt_poses[gt_flag].timestamp:
                gt_flag = gt_flag + 1
            self.gt_poses_selected.append(pose.merge(es_timestamp,self.gt_poses[gt_flag],self.gt_poses[gt_flag],interpolation_flag))
    
    def computeError(self):
        if(len(self.es_poses_selected)!=len(self.gt_poses_selected)):
            print("【Error】The number of selected estimated and ground truth poses is not equal.")
            return
        
        RMSE = 0.0
        num_pose = len(self.es_poses_selected)
        for i in range(num_pose-1):
            # Ei = (Q1^-1 · Q2)·(P1^-1 · P2)
            P1 = self.es_poses_selected[i].transformation()
            P1 = self.es_poses_selected[i+1].transformation()
            Q1 = self.gt_poses_selected[i].transformation()
            Q1 = self.gt_poses_selected[i+1].transformation()
            

        
        # RMSE = sqrt{ SUM[ (trans(Ei))^2 ] / m }
    

def main(args=None):
    time_registration = TimeRegistration()
    time_registration.register()
    time_registration.computeError()

    # 完成时间同步
    # 计算相对位姿误差


if __name__ == '__main__':
    main()