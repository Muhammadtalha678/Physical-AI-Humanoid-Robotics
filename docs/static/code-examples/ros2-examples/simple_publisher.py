#!/usr/bin/env python3

"""
Simple ROS 2 Publisher Example
This node publishes messages to a topic at a regular interval.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')

        # Create a publisher
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create a timer to publish messages at 2Hz
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter for messages
        self.i = 0

        self.get_logger().info('Simple Publisher Node Initialized')

    def timer_callback(self):
        """Callback function that publishes messages"""
        msg = String()
        msg.data = f'Hello World: {self.i}'

        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

        self.i += 1


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    simple_publisher = SimplePublisher()

    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        simple_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()