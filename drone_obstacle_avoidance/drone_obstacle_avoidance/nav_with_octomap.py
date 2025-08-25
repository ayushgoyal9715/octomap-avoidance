#!/usr/bin/env python3

import asyncio
import random
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

import octomap
from octomap_msgs.msg import Octomap


class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')
        self.get_logger().info("ObstacleAvoider node initialized (Depth Camera + Octomap).")

        # CV bridge for depth image
        self.bridge = CvBridge()
        self.safe_distance = 1.25  # meters
        self.min_dist = float('inf')
        self.state = "FORWARD"  # FSM state

        # Camera intrinsics (default, adjust if needed)
        self.fx, self.fy = 524, 524
        self.cx, self.cy = 316.8, 238.5

        # Octomap setup
        self.octree = octomap.OcTree(0.1)  # resolution: 10cm
        self.frame_id = "map"

        # ROS publishers/subscribers
        self.octomap_pub = self.create_publisher(Octomap, "map_out", 10)
        self.create_subscription(Image, '/front_cam/depth', self.depth_callback, 10)
        self.create_subscription(Odometry, '/drone/odom', self.odom_callback, 10)

        # State vars
        self.depth_img = None
        self.current_pose = None

    def depth_callback(self, msg: Image):
        """Depth image callback."""
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        h, w = self.depth_img.shape
        center_region = self.depth_img[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        self.min_dist = np.nanmin(center_region)

        # Update Octomap if pose available
        if self.current_pose is not None:
            self.update_octomap()

    def odom_callback(self, msg: Odometry):
        """Odometry callback for pose."""
        self.current_pose = msg.pose.pose

    def update_octomap(self):
        """Generate Octomap rays from depth image and pose."""
        origin = octomap.point3d(
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z,
        )

        for v in range(0, self.depth_img.shape[0], 4):  # skip pixels for efficiency
            for u in range(0, self.depth_img.shape[1], 4):
                d = self.depth_img[v, u]
                if np.isnan(d) or d <= 0:
                    continue

                # Back-project pixel to 3D (camera frame)
                x = (u - self.cx) * d / self.fx
                y = (v - self.cy) * d / self.fy
                z = d

                # Convert camera coords to world frame (assumes forward-facing camera)
                wx = origin.x() + x
                wy = origin.y() + y
                wz = origin.z() - z

                endpoint = octomap.point3d(wx, wy, wz)
                self.octree.insertRay(origin, endpoint)

        # Publish Octomap
        msg = Octomap()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.binary = True
        msg.id = "OcTree"
        msg.data = list(self.octree.writeBinaryConst().read())
        self.octomap_pub.publish(msg)

    def compute_velocity_command(self):
        """Finite State Machine for Obstacle Avoidance."""
        if self.state == "FORWARD":
            if self.min_dist < self.safe_distance:
                self.get_logger().warn(f"Obstacle {self.min_dist:.2f} m ahead! Stopping.")
                self.state = "STOP"
                return 0.0, 0.0
            else:
                return 0.8, 0.0  # forward velocity

        elif self.state == "STOP":
            self.state = random.choice(["TURN_RIGHT"])
            self.get_logger().info(f"Switching to {self.state} to avoid obstacle.")
            return 0.0, 0.0

        elif self.state == "TURN_RIGHT":
            if self.min_dist < self.safe_distance:
                yaw_rate = -0.5
                self.get_logger().info("Turning RIGHT...")
                return 0.0, yaw_rate
            else:
                self.get_logger().info("Path clear! Moving forward.")
                self.state = "FORWARD"
                return 0.8, 0.0

        return 0.0, 0.0


async def drone_control(node: ObstacleAvoider):
    """Drone control loop using MAVSDK."""
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    node.get_logger().info("Connecting to drone...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            node.get_logger().info("Drone connected!")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            node.get_logger().info("Position lock acquired.")
            break

    await drone.action.arm()
    node.get_logger().info("Arming drone...")
    await drone.action.takeoff()
    node.get_logger().info("Taking off...")
    await asyncio.sleep(28)  # allow climb to height

    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        node.get_logger().info("Offboard started.")
    except OffboardError as e:
        node.get_logger().error(f"Offboard failed: {e._result.result_str}")
        return

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
        forward, yaw = node.compute_velocity_command()
        try:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, 0.0, 0.0, np.rad2deg(yaw))
            )
        except OffboardError as e:
            node.get_logger().error(f"Offboard failed: {e._result.result_str}")
            break
        await asyncio.sleep(0.1)

    await drone.offboard.stop()
    await drone.action.land()
    await drone.action.disarm()
    node.get_logger().info("Landed & disarmed.")


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoider()
    try:
        asyncio.run(drone_control(node))
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt. Landing drone.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
