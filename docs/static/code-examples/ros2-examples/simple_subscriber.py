#!/usr/bin/env python3

"""
Simple ROS 2 Subscriber Example
This node subscribes to messages from a topic and logs them.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    def __init__(self):
        super().__init__('simple_subscriber')

        # Create a subscriber
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        self.get_logger().info('Simple Subscriber Node Initialized')

    def listener_callback(self, msg):
        """Callback function that processes received messages"""
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    simple_subscriber = SimpleSubscriber()

    try:
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        simple_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()