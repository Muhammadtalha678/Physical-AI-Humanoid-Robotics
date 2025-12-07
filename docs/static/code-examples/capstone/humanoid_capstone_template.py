#!/usr/bin/env python3

"""
Humanoid Capstone Project Template
This is a template for the capstone project integrating speech, planning,
navigation, perception, and manipulation for an autonomous humanoid robot.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import numpy as np


class HumanoidCapstoneNode(Node):
    """
    Capstone project node that integrates speech, planning, navigation,
    perception, and manipulation for a humanoid robot.
    """
    def __init__(self):
        super().__init__('humanoid_capstone_node')

        # Subscribers for different sensor modalities
        self.speech_sub = self.create_subscription(
            String,
            '/speech_recognition/text',
            self.speech_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.camera_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Publishers for navigation and manipulation
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)

        # Internal state
        self.current_task = "idle"
        self.navigation_goal = None
        self.manipulation_target = None
        self.robot_pose = None

        self.get_logger().info("Humanoid Capstone Node Initialized")

    def speech_callback(self, msg):
        """Process speech commands to understand tasks"""
        command = msg.data.lower()
        self.get_logger().info(f"Received speech command: {command}")

        # Simple command parsing - in practice this would use NLP
        if "go to" in command or "navigate to" in command:
            self.parse_navigation_command(command)
        elif "pick up" in command or "grasp" in command:
            self.parse_manipulation_command(command)
        elif "stop" in command:
            self.stop_robot()
        else:
            self.get_logger().info(f"Unknown command: {command}")

    def parse_navigation_command(self, command):
        """Parse navigation commands"""
        # Simple location extraction - would be more sophisticated in real implementation
        if "kitchen" in command:
            self.navigation_goal = self.get_location_pose("kitchen")
            self.start_navigation()
        elif "living room" in command:
            self.navigation_goal = self.get_location_pose("living_room")
            self.start_navigation()
        else:
            self.get_logger().info(f"Unknown location in command: {command}")

    def parse_manipulation_command(self, command):
        """Parse manipulation commands"""
        # Simple object extraction
        if "red cup" in command:
            self.manipulation_target = "red_cup"
            self.start_perception_and_manipulation()
        elif "blue box" in command:
            self.manipulation_target = "blue_box"
            self.start_perception_and_manipulation()

    def laser_callback(self, msg):
        """Process laser scan data for navigation and obstacle detection"""
        # Store laser data for navigation planning
        pass

    def camera_callback(self, msg):
        """Process camera data for perception"""
        # Store image data for object detection
        pass

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.robot_pose = msg.pose.pose

    def get_location_pose(self, location_name):
        """Get predefined poses for known locations"""
        # In a real system, this would come from a map
        locations = {
            "kitchen": PoseStamped(),
            "living_room": PoseStamped(),
            "bedroom": PoseStamped(),
        }

        # Set dummy poses for example
        if location_name in locations:
            pose_stamped = PoseStamped()
            if location_name == "kitchen":
                pose_stamped.pose.position.x = 5.0
                pose_stamped.pose.position.y = 2.0
            elif location_name == "living_room":
                pose_stamped.pose.position.x = 0.0
                pose_stamped.pose.position.y = 0.0
            elif location_name == "bedroom":
                pose_stamped.pose.position.x = -3.0
                pose_stamped.pose.position.y = 4.0

            return pose_stamped

        return None

    def start_navigation(self):
        """Start navigation to the current goal"""
        if self.navigation_goal:
            self.get_logger().info(f"Navigating to: {self.navigation_goal.pose.position}")
            # Publish navigation goal
            self.goal_pub.publish(self.navigation_goal)
            self.current_task = "navigating"

    def start_perception_and_manipulation(self):
        """Start perception and manipulation sequence"""
        self.get_logger().info(f"Looking for object: {self.manipulation_target}")
        # Perception and manipulation would be triggered here
        self.current_task = "perceiving_manipulating"

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)
        self.current_task = "idle"
        self.get_logger().info("Robot stopped")

    def run_sequence(self):
        """Run the full capstone sequence"""
        self.get_logger().info("Starting capstone sequence...")

        # This would orchestrate the full autonomous pipeline
        # 1. Listen for speech command
        # 2. Plan navigation
        # 3. Navigate to location
        # 4. Perceive target object
        # 5. Plan and execute manipulation
        # 6. Return or await next command


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    capstone_node = HumanoidCapstoneNode()

    try:
        # In a real implementation, you might want to run a sequence
        # capstone_node.run_sequence()
        rclpy.spin(capstone_node)
    except KeyboardInterrupt:
        pass
    finally:
        capstone_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()