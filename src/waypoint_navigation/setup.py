from setuptools import setup

package_name = 'waypoint_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Muhammad Adeel Zahid',
    maintainer_email='adeel.m.zahid@gmail.com',
    description='Waypoint following with pure pursuit on odometry.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'follower_node = waypoint_navigation.follower_node:main',
            'nav_goal = waypoint_navigation.nav_goal:main',
        ],
    },
)
