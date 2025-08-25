from setuptools import find_packages, setup

package_name = 'drone_obstacle_avoidance'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
           'avoider = drone_obstacle_avoidance.drone_control:main',
           'lidar_obstacle_avoider = drone_obstacle_avoidance.lidar_obstacle_avoider:main',
           'drone_control_with_octomap = drone_obstacle_avoidance.nav_with_octomap:main',
           'odom_to_pose = drone_obstacle_avoidance.odom_to_pose:main', 
        ],
    },
)
