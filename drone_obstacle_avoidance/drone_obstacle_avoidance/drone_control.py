#!/usr/bin/env python3

import asyncio
import random
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed


class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')
        self.get_logger().info("ObstacleAvoider node initialized (Depth Camera).")

        self.bridge = CvBridge()
        self.safe_distance = 1.25  # meters
        self.min_dist = float('inf')

        # FSM states: FORWARD, STOP, TURN_LEFT, TURN_RIGHT
        self.state = "FORWARD"

        # Subscribe to depth camera
        self.create_subscription(Image, '/front_cam/depth', self.depth_callback, 10)

    def depth_callback(self, msg: Image):
        """Process depth image and update min_dist."""
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        h, w = depth_img.shape

        # Central region for obstacle detection
        center_region = depth_img[h//3: 2*h//3, w//3: 2*w//3]
        self.min_dist = np.nanmin(center_region)

    def compute_velocity_command(self):
        """Finite State Machine: Forward → Stop → Turn → Forward"""
        if self.state == "FORWARD":
            if self.min_dist < self.safe_distance:
                self.get_logger().warn(f"Obstacle {self.min_dist:.2f} m ahead! Stopping.")
                self.state = "STOP"
                return 0.0, 0.0  # Immediate stop
            else:
                return 0.8, 0.0  # Move forward

        elif self.state == "STOP":
            # Pick random turn direction after stopping
            self.state = random.choice([ "TURN_RIGHT"])
            self.get_logger().info(f"Switching to {self.state} to avoid obstacle.")
            return 0.0, 0.0  # Stop for one cycle before turning

        elif self.state in [ "TURN_RIGHT"]:
            if self.min_dist < self.safe_distance:
                yaw_rate = 0.5 if self.state == "TURN_LEFT" else -0.5
                self.get_logger().info(f"Turning {self.state.split('_')[1]}...")
                return 0.0, yaw_rate  # Rotate in place until clear
            else:
                self.get_logger().info("Path clear! Moving forward.")
                self.state = "FORWARD"
                return 0.8, 0.0  # Resume forward

        return 0.0, 0.0


async def drone_control(node: ObstacleAvoider):
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14553")
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
    await asyncio.sleep(28)

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
