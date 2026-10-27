from setuptools import setup

package_name = 'collision_monitor'

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
    description='Distance / time-to-collision warnings and emergency stop.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'monitor_node = collision_monitor.monitor_node:main',
        ],
    },
)
