from glob import glob

from setuptools import setup

package_name = "aruco_localization"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Assignment 4: camera, ArUco detection, odometry and particle filter localization for LeKiwi",
    license="GPL-3.0-only",
    entry_points={
        "console_scripts": [
            "camera_node = aruco_localization.camera_node:main",
            "aruco_detector = aruco_localization.aruco_detector:main",
            "odometry_node = aruco_localization.odometry_node:main",
            "map_node = aruco_localization.map_node:main",
            "localization_node = aruco_localization.localization_node:main",
            "keyboard_teleop = aruco_localization.keyboard_teleop:main",
        ],
    },
)
