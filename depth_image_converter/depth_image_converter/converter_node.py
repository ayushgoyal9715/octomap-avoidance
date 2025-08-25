import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthImageConverter(Node):
    def __init__(self):
        super().__init__('depth_image_converter')
        self.subscription = self.create_subscription(
            Image,
            '/front_cam/depth',  # Input topic
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(
            Image,
            '/front_cam/depth_converted',  # Output topic
            10)
        self.bridge = CvBridge()
        self.get_logger().info('Depth image converter node has been started.')

    def listener_callback(self, msg):
        if msg.encoding == '32FC1':
            try:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

                # Handle NaN and inf values by replacing them with 0
                cv_image = np.nan_to_num(cv_image, nan=0.0, posinf=0.0, neginf=0.0)

                # Convert from meters to millimeters and change data type to 16-bit unsigned integer
                cv_image_mm = (cv_image * 1000).astype(np.uint16)

                # Convert OpenCV image back to ROS Image message
                converted_msg = self.bridge.cv2_to_imgmsg(cv_image_mm, encoding='mono16')

                # Copy header information
                converted_msg.header = msg.header

                # Publish the converted message
                self.publisher.publish(converted_msg)

            except Exception as e:
                self.get_logger().error(f'Failed to convert image: {e}')

def main(args=None):
    rclpy.init(args=args)
    depth_image_converter = DepthImageConverter()
    rclpy.spin(depth_image_converter)
    depth_image_converter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
