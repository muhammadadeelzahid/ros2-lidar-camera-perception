from setuptools import setup

package_name = 'perception_tools'

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
    description='Dataset, calibration, scenario and evaluation tools.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'auto_labeler = perception_tools.auto_labeler:main',
            'dataset_export = perception_tools.dataset_export:main',
            'eval_metrics = perception_tools.eval_metrics:main',
            'generate_world = perception_tools.generate_world:main',
            'calibrate_camera = perception_tools.calibrate_camera:main',
            'calibrate_imu = perception_tools.calibrate_imu:main',
            'stability_scenarios = perception_tools.stability_scenarios:main',
            'lidar_camera_overlay = perception_tools.lidar_camera_overlay:main',
            'wander = perception_tools.wander:main',
            'agent_mover = perception_tools.agent_mover:main',
            'eval_detector = perception_tools.eval_detector:main',
        ],
    },
)
