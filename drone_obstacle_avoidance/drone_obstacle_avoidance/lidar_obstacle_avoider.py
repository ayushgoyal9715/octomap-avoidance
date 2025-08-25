import asyncio
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed


class LidarObstacleAvoider(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_avoider')
        self.get_logger().info("LidarObstacleAvoider node initialized (LiDAR).")

        self.safe_distance = 1.25  # meters (can be as low as 0.25 now)
        self.buffer_distance = 0.2  # Extra buffer for latency compensation
        self.min_dist = float('inf')

        # LiDAR regions
        self.front_region = []
        self.left_region = []
        self.right_region = []

        # FSM states: FORWARD, EVALUATE, TURN_LEFT, TURN_RIGHT
        self.state = "FORWARD"

        # Subscribe to LiDAR scan topic
        self.create_subscription(LaserScan, '/lidar/scan', self.lidar_callback, 10)

    def lidar_callback(self, msg: LaserScan):
        """Process LiDAR scan into regions."""
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), np.nan, ranges)  # Replace inf with NaN

        num_points = len(ranges)
        center = num_points // 2

        # Define regions: front ±20°, left 60°–120°, right -120°–-60°
        front_fov = int(num_points * (20.0 / 180.0))
        side_fov = int(num_points * (60.0 / 180.0))

        self.front_region = ranges[center - front_fov:center + front_fov]
        self.left_region = ranges[center + front_fov:center + front_fov + side_fov]
        self.right_region = ranges[center - front_fov - side_fov:center - front_fov]

        self.min_dist = np.nanmin(self.front_region)

    def compute_velocity_command(self):
        """FSM with dynamic speed scaling and intelligent turning."""
        if self.state == "FORWARD":
            # Slow down early if close to an obstacle
            if self.min_dist < self.safe_distance + self.buffer_distance:
                self.get_logger().warn(f"Obstacle {self.min_dist:.2f} m ahead! Slowing down.")

                if self.min_dist < self.safe_distance:
                    self.state = "EVALUATE"
                    return 0.0, 0.0  # Stop before turn
                else:
                    # Dynamic speed scaling (slower when closer)
                    speed = max(0.15, (self.min_dist - 0.1))  # Minimum 0.15 m/s
                    return speed, 0.0
            else:
                return 0.8, 0.0  # Full speed forward

        elif self.state == "EVALUATE":
            # Compare clearance left vs right
            left_clearance = np.nanmean(self.left_region) if len(self.left_region) > 0 else 0
            right_clearance = np.nanmean(self.right_region) if len(self.right_region) > 0 else 0

            if left_clearance > right_clearance:
                self.state = "TURN_LEFT"
                self.get_logger().info(f"Turning LEFT (clearance: {left_clearance:.2f}m > {right_clearance:.2f}m).")
            else:
                self.state = "TURN_RIGHT"
                self.get_logger().info(f"Turning RIGHT (clearance: {right_clearance:.2f}m >= {left_clearance:.2f}m).")
            return 0.0, 0.0  # Stop before turning

        elif self.state in ["TURN_LEFT", "TURN_RIGHT"]:
            if self.min_dist < self.safe_distance:
                yaw_rate = 0.5 if self.state == "TURN_LEFT" else -0.5
                self.get_logger().info(f"Turning {self.state.split('_')[1]}...")
                return 0.0, yaw_rate  # Rotate in place
            else:
                self.get_logger().info("Path clear! Moving forward.")
                self.state = "FORWARD"
                return 0.8, 0.0

        return 0.0, 0.0


async def drone_control(node: LidarObstacleAvoider):
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

    # Set takeoff height before takeoff
    await drone.action.set_takeoff_altitude(5.0)  # ✅ Controlled takeoff height
    await drone.action.arm()
    node.get_logger().info("Arming drone...")
    await drone.action.set_takeoff_altitude(1.5)  
    await drone.action.takeoff()
    node.get_logger().info("Taking off...")
    await asyncio.sleep(10)  # Allow stabilization at takeoff height

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
    node = LidarObstacleAvoider()
    try:
        asyncio.run(drone_control(node))
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt. Landing drone.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
