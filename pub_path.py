from calendar import c
from os import path
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from math import sin, cos, pi
import numpy as np
import tf2_ros


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('VINS_path')
        self.publisher_ = self.create_publisher(Path, 'VINS', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.msg = TransformStamped()
        self.msg.header.frame_id = "map"
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        self.arr = []
        self.i = 0
        self.ct = 0
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        with open('/home/jiangtao.li/Downloads/result_V101.txt') as f:
            contents = f.readlines()
            for line in contents:
                arr = []
                line = line.strip()
                raw = line.split()
                for elem in raw:
                    arr.append(float(elem))
                self.arr.append(arr)    

    def timer_callback(self):
        
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.arr[self.i][1]
        pose.pose.position.y = self.arr[self.i][2]
        pose.pose.position.z = self.arr[self.i][3]

        pose.pose.orientation.x = self.arr[self.i][4]
        pose.pose.orientation.y = self.arr[self.i][5]
        pose.pose.orientation.z = self.arr[self.i][6]
        pose.pose.orientation.w = self.arr[self.i][7]
        
        self.msg.child_frame_id = "chassis"
        self.msg.transform.translation.x = self.arr[self.i][1]
        self.msg.transform.translation.y = self.arr[self.i][2]
        self.msg.transform.translation.z = self.arr[self.i][3]
        self.msg.transform.rotation.x = self.arr[self.i][4]
        self.msg.transform.rotation.y = self.arr[self.i][5]
        self.msg.transform.rotation.z = self.arr[self.i][6]
        self.msg.transform.rotation.w = self.arr[self.i][7]
            
        self.path_msg.poses.append(pose)
        if self.i > 900: 
            self.i = 0

        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.msg)
        self.publisher_.publish(self.path_msg)
        
        self.i += 1
        print(self.i)
        



class MinimalPublisherSync(Node):

    def __init__(self):
        super().__init__('VINS_path')
        self.publisher_ = self.create_publisher(Path, 'VINS', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.msg = TransformStamped()
        self.msg.header.frame_id = "map"
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        self._last_pos_ = [0, 0, 0]
        self.ct = 0
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def timer_callback(self):
        
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        with open('/home/jiangtao.li/workplace/vins-fusion-gpu-no-ros/vins_estimator/build/VIO.txt') as f:
            contents = f.readlines()
            if len(contents) < 3: 
                self.path_msg.poses.clear()
                return
            line = contents[-1]
            arr = []
            line = line.strip()
            raw = line.split()
            for elem in raw:
                arr.append(float(elem))
        pose.pose.position.x = arr[1]
        pose.pose.position.y = arr[2]
        pose.pose.position.z = arr[3]

        pose.pose.orientation.x = arr[4]
        pose.pose.orientation.y = arr[5]
        pose.pose.orientation.z = arr[6]
        pose.pose.orientation.w = arr[7]
        
        self.msg.child_frame_id = "chassis"
        self.msg.transform.translation.x = arr[1]
        self.msg.transform.translation.y = arr[2]
        self.msg.transform.translation.z = arr[3]
        self.msg.transform.rotation.x = arr[4]
        self.msg.transform.rotation.y = arr[5]
        self.msg.transform.rotation.z = arr[6]
        self.msg.transform.rotation.w = arr[7]
        if not (self._last_pos_[0] == arr[1] and self._last_pos_[1] == arr[2] and self._last_pos_[2] == arr[3]):
            self.path_msg.poses.append(pose)
        
        self._last_pos_ = [arr[1], arr[2], arr[3]]

        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.msg)
        self.publisher_.publish(self.path_msg)
        


class GTPublisher(Node):
    
    def __init__(self):
        super().__init__('VINS_gt_path')
        self.publisher_ = self.create_publisher(Path, 'VINS_GT', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
 
        gt = np.fromfile('/home/jiangtao.li/Downloads/gt.bin')
        self.gt = gt.reshape((1035, 8))
        self.gt_msg = TransformStamped()
        self.gt_msg.header.frame_id = "map"
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = "map"
        self.i = 0
        self.ct = 0
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def timer_callback(self):
        
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.gt[self.i][1]
        pose.pose.position.y = self.gt[self.i][2]
        pose.pose.position.z = self.gt[self.i][3]

        pose.pose.orientation.x = self.gt[self.i][4]
        pose.pose.orientation.y = self.gt[self.i][5]
        pose.pose.orientation.z = self.gt[self.i][6]
        pose.pose.orientation.w = self.gt[self.i][7]
        
        self.gt_msg.child_frame_id = "chassis"
        self.gt_msg.transform.translation.x = self.gt[self.i][1]
        self.gt_msg.transform.translation.y = self.gt[self.i][2]
        self.gt_msg.transform.translation.z = self.gt[self.i][3]
        self.gt_msg.transform.rotation.x = self.gt[self.i][4]
        self.gt_msg.transform.rotation.y = self.gt[self.i][5]
        self.gt_msg.transform.rotation.z = self.gt[self.i][6]
        self.gt_msg.transform.rotation.w = self.gt[self.i][7]

        self.i += 1
            
        self.gt_path_msg.poses.append(pose)
        if self.i > 900: 
            self.i = 0

        self.gt_msg.header.stamp = self.get_clock().now().to_msg()
        self.gt_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.gt_msg)
        self.publisher_.publish(self.gt_path_msg)
        print("GT")
        print(self.i)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisherSync()
    gt_publisher = GTPublisher()
    rclpy.spin(minimal_publisher)
    # rclpy.spin(gt_publisher)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    gt_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()