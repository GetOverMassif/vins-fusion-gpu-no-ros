#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Compute the error between the estimated poses and the ground true"""

from os import path
# from geometry_msgs.msg import PoseStamped, TransformStamped
import csv
from turtle import pos

estimated_poses_path = "/home/lj/Documents/vins-fusion-gpu-no-ros/vins_estimator/build/VIO.txt"
ground_trueth_path = "/home/lj/data/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"

TimeStampEstimated = 0
TimeStampGroundTruth = 1

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



class TimeRegistration():
    # global estimated_poses_path,ground_trueth_path
    def __init__(self):
        estimated_poses = []
        ground_trueth = []
        # 读取估计位姿
        with open(estimated_poses_path) as f:
            count = 0
            contents = f.readlines()

            while (len(contents[count].split(" "))!=12):
                count = count+1
            contents = contents[count:]
            for line in contents:
                estimated_poses.append(pose(line,TimeStampEstimated))
            print("Num of estimated_poses:",len(estimated_poses))
            print(estimated_poses[0].timestamp)

        # 读取ground truth
        with open(ground_trueth_path) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=' ', quotechar='|')
            # print(type(csv_reader))
            # print(",".join(csv_reader[0]))
            count2 = 0
            for row in csv_reader:
                if(count2==0):
                    count2 = count2+1
                    continue
                ground_trueth.append(pose((" ".join(row)).replace(","," "),TimeStampGroundTruth))
            print("Num of ground_trueth:",len(ground_trueth))
            print(ground_trueth[0].timestamp)
            #     print(type(row[0]))
            #     print(','.join(row))



def main(args=None):
    time_registration = TimeRegistration()

    # 完成时间同步
    # 计算相对位姿误差


if __name__ == '__main__':
    main()