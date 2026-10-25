"""Drive the robot's base with the keyboard.

Run it in its own terminal (it reads the keys typed there):

    ros2 run aruco_localization keyboard_teleop

Keys:
    w / s     forward / backward
    a / d     left / right (sideways: the base is omnidirectional)
    q / e     turn left / turn right
    space     stop
    + / -     faster / slower
    Ctrl+C    quit (the robot stops)

Publishes:
    base_commands  std_msgs/Float64MultiArray, 3 values: forward (m/s), left (m/s), turn (deg/s), ten times
                   a second. The robot stops by itself half a second after the last command.
Parameters:
    speed       starting driving speed in m/s (default 0.1)
    turn_speed  starting turning speed in deg/s (default 30)
"""
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# key: (forward, left, turn) as multiples of the speeds
KEYS = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0), "q": (0, 0, 1), "e": (0, 0, -1), " ": (0, 0, 0)}


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")
        self.declare_parameter("speed", 0.1)
        self.declare_parameter("turn_speed", 30.0)
        self.speed = self.get_parameter("speed").value
        self.turn_speed = self.get_parameter("turn_speed").value
        self.motion = (0, 0, 0)
        self.publisher = self.create_publisher(Float64MultiArray, "base_commands", 10)
        self.create_timer(0.1, self.on_timer)
        print(__doc__.split("Publishes:")[0].split("Keys:")[1])
        self.show()

    def show(self):
        print(f"\rspeed {self.speed:.2f} m/s, turn {self.turn_speed:.0f} deg/s, moving {self.motion}      ", end="")

    def on_timer(self):
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if key in KEYS:
                self.motion = KEYS[key]
            elif key == "+":
                self.speed, self.turn_speed = self.speed * 1.25, self.turn_speed * 1.25
            elif key == "-":
                self.speed, self.turn_speed = self.speed / 1.25, self.turn_speed / 1.25
            self.show()
        forward, left, turn = self.motion
        self.publisher.publish(Float64MultiArray(data=[forward * self.speed, left * self.speed, turn * self.turn_speed]))

    def stop(self):
        self.publisher.publish(Float64MultiArray(data=[0.0, 0.0, 0.0]))


def main():
    if not sys.stdin.isatty():
        sys.exit("keyboard_teleop must run in a terminal: ros2 run aruco_localization keyboard_teleop")
    settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())   # deliver each key at once, without waiting for Enter
    rclpy.init()
    node = KeyboardTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
