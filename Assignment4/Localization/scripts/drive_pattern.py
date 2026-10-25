"""Drive the base along a fixed pattern, so that an experiment can be repeated exactly.

Usage (in a sourced terminal, while the launch file runs):

    python drive_pattern.py square [--size 1.0] [--speed 0.15] [--laps 1]
    python drive_pattern.py circle [--size 0.8] [--speed 0.15] [--laps 1]
    python drive_pattern.py spin   [--laps 1]          # turn on the spot
    python drive_pattern.py still  [--time 30]         # stand still (sends zero commands)

Patterns:
    square  drive forward `size` metres, turn left 90 degrees, four times per lap
    circle  drive on a circle of diameter `size` metres, counter-clockwise
    spin    turn 360 degrees on the spot per lap
    still   send "stop" for `time` seconds

The pattern is driven by time (speed x time), not by feedback, so the robot ends up only roughly where
it started. That is fine: the point is to make the same commands every time.

Publishes:
    base_commands  std_msgs/Float64MultiArray: forward (m/s), left (m/s), turn (deg/s), 20 times a second
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

RATE = 20.0          # commands per second
TURN_SPEED = 30.0    # deg/s for the turns of the square and the spin


def segments(args):
    """The pattern as a list of (seconds, [forward, left, turn])."""
    if args.pattern == "square":
        side = (args.size / args.speed, [args.speed, 0.0, 0.0])
        turn = (90.0 / TURN_SPEED, [0.0, 0.0, TURN_SPEED])
        return [side, turn] * 4 * args.laps
    if args.pattern == "circle":
        circumference = math.pi * args.size
        seconds = circumference / args.speed
        return [(seconds * args.laps, [args.speed, 0.0, 360.0 / seconds])]
    if args.pattern == "spin":
        return [(360.0 / TURN_SPEED * args.laps, [0.0, 0.0, TURN_SPEED])]
    return [(args.time, [0.0, 0.0, 0.0])]


class PatternDriver(Node):
    def __init__(self, plan):
        super().__init__("drive_pattern")
        self.publisher = self.create_publisher(Float64MultiArray, "base_commands", 10)
        self.plan = plan
        self.ticks = 0
        self.total = sum(seconds for seconds, _ in plan)
        self.get_logger().info(f"Driving for {self.total:.1f} s.")
        self.create_timer(1.0 / RATE, self.on_timer)

    def on_timer(self):
        t = self.ticks / RATE
        self.ticks += 1
        for seconds, command in self.plan:
            if t < seconds:
                self.publisher.publish(Float64MultiArray(data=command))
                return
            t -= seconds
        self.publisher.publish(Float64MultiArray(data=[0.0, 0.0, 0.0]))
        self.get_logger().info("Pattern done.")
        raise SystemExit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", choices=["square", "circle", "spin", "still"])
    parser.add_argument("--size", type=float, default=1.0, help="side of the square / diameter of the circle (m)")
    parser.add_argument("--speed", type=float, default=0.15, help="driving speed (m/s)")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--time", type=float, default=30.0, help="seconds, for the pattern still")
    args = parser.parse_args()
    rclpy.init()
    node = PatternDriver(segments(args))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.publisher.publish(Float64MultiArray(data=[0.0, 0.0, 0.0]))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
