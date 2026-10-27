from setuptools import setup

package_name = 'camera_detection'

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
    description='2D image object detection and classification.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node = camera_detection.yolo_node:main',
        ],
    },
)
