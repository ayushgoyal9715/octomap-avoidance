#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class OdomToPose(Node):
    def __init__(self):
        super().__init__('odom_to_pose')
        self.create_subscription(Odometry, '/drone/odom', self.odom_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/drone/pose', 10)
        self.get_logger().info("Odom to Pose converter node started.")

    def odom_callback(self, msg: Odometry):
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OdomToPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
