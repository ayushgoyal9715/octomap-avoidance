from launch.substitutions import LaunchConfiguration, PythonExpression

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'full_sensor_topic',
            default_value=['/front_cam'],
            description='Base for topic (and node) names'),
        DeclareLaunchArgument(
            'range_max',
            default_value='19.0',
            description='Max range of depth sensor'),
        DeclareLaunchArgument(
            'use_quiet_nan',
            default_value='false',
            description='Use quiet NaN instead of range_max'),
        DeclareLaunchArgument(
            'depth_image_topic',
            default_value='/front_cam/depth',
            description='Depth image topic'),
        DeclareLaunchArgument(
            'rgb_image_topic',
            default_value='/camera/rgb',
            description='Colorize the point cloud from external RGB image topic'),
        Node(
            package="depthimage_to_pointcloud2",
            executable="depthimage_to_pointcloud2_node",
            output='screen',
            name='rgbd_to_pointcloud',
            parameters=[
                {'range_max': LaunchConfiguration('range_max'),
                 'use_quiet_nan': LaunchConfiguration('use_quiet_nan'),
                 'colorful': PythonExpression(["'true' if '", LaunchConfiguration('rgb_image_topic'), "' else 'false'"])}
            ],
            remappings=[
            ("image", "/front_cam/image"),
            ("depth", "/front_cam/depth"),
            ("depth_camera_info", "/front_cam/info"),
            ("pointcloud2", "/camera/pointcloud")
            ]
        )
    ])
