---
sidebar_position: 3
title: "Building ROS 2 Packages with Python"
---

# Building ROS 2 Packages with Python

This section covers how to create, structure, and build ROS 2 packages using Python. Python is one of the most popular languages for ROS 2 development due to its ease of use and rapid prototyping capabilities.

## ROS 2 Package Structure

### Basic Package Layout
A typical ROS 2 Python package follows this structure:
```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++ components
├── package.xml             # Package metadata
├── setup.cfg               # Python installation configuration
├── setup.py                # Python package setup script
├── resource/               # Resource files
├── launch/                 # Launch files
├── config/                 # Configuration files
├── test/                   # Test files
├── my_robot_package/       # Python package directory
│   ├── __init__.py         # Python package initialization
│   ├── my_node.py          # Python node implementation
│   └── my_module.py        # Additional Python modules
└── README.md               # Documentation
```

### Package.xml File
The `package.xml` file contains metadata about the package:
- Package name, version, and description
- Maintainer information
- License information
- Dependencies and build tools
- Exported interfaces

Example package.xml:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 Python package</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Setting Up Python Packages

### setup.py Configuration
The `setup.py` file defines how the Python package is installed:
```python
from setuptools import setup
from glob import glob
import os

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Example ROS 2 Python package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_robot_package.my_node:main',
        ],
    },
)
```

### setup.cfg Configuration
The `setup.cfg` file contains installation configuration:
```
[develop]
script-dir=$base/lib/my_robot_package

[install]
install-scripts=$base/lib/my_robot_package
```

## Creating Python Nodes

### Basic Node Structure
A basic ROS 2 Python node follows this pattern:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.publisher = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    my_node = MyNode()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Publisher Implementation
Creating publishers in Python:
```python
# Create publisher with QoS profile
self.publisher = self.create_publisher(
    msg_type=String,
    topic='topic_name',
    qos_profile=10  # or use QoSProfile object
)

# Publish messages
msg = String()
msg.data = 'message content'
self.publisher.publish(msg)
```

### Subscriber Implementation
Creating subscribers in Python:
```python
# Create subscriber with callback
self.subscriber = self.create_subscription(
    msg_type=String,
    topic='topic_name',
    callback=self.subscription_callback,
    qos_profile=10
)

def subscription_callback(self, msg):
    self.get_logger().info(f'Received: {msg.data}')
```

### Service Implementation
Creating services in Python:
```python
from example_interfaces.srv import AddTwoInts

def __init__(self):
    super().__init__('service_server')
    self.srv = self.create_service(
        AddTwoInts,
        'add_two_ints',
        self.add_two_ints_callback
    )

def add_two_ints_callback(self, request, response):
    response.sum = request.a + request.b
    self.get_logger().info(f'Returning {response.sum}')
    return response
```

### Action Implementation
Creating actions in Python:
```python
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Building and Running Python Packages

### Creating a Package
```bash
ros2 pkg create --build-type ament_python my_robot_package
```

### Building the Package
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

### Running Nodes
```bash
# Run node directly
ros2 run my_robot_package my_node

# Run with parameters
ros2 run my_robot_package my_node --ros-args --param my_param:=value
```

## Launch Files for Python Packages

### Python Launch Files
ROS 2 supports launch files written in Python:
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='my_node',
            name='my_node',
            parameters=[
                {'param1': 'value1'},
                {'param2': 42}
            ],
            remappings=[
                ('original_topic', 'new_topic')
            ]
        )
    ])
```

## Testing Python Packages

### Unit Testing
Python packages can be tested using pytest:
```python
import pytest
import rclpy
from my_robot_package.my_node import MyNode

def test_node_creation():
    rclpy.init()
    try:
        node = MyNode()
        assert node is not None
    finally:
        rclpy.shutdown()
```

### Launch Testing
Testing launch files and integration:
```python
import launch
from launch_ros.actions import Node
import launch_testing.actions

def generate_test_description():
    node = Node(
        package='my_robot_package',
        executable='my_node',
        name='test_node'
    )

    return launch.LaunchDescription([
        node,
        launch_testing.actions.ReadyToTest()
    ])
```

## Best Practices

### Code Organization
- Use proper Python packaging with `__init__.py` files
- Separate concerns into different modules
- Follow PEP 8 style guidelines
- Use meaningful names for variables and functions

### Error Handling
- Use try-catch blocks where appropriate
- Log errors with appropriate severity levels
- Handle ROS-specific exceptions
- Gracefully handle shutdown scenarios

### Performance Considerations
- Minimize message copying in callbacks
- Use appropriate QoS settings
- Consider threading for CPU-intensive operations
- Profile code to identify bottlenecks

## Learning Objectives

After completing this section, you should be able to:
- Create and structure ROS 2 packages using Python
- Implement nodes with publishers, subscribers, services, and actions
- Configure package metadata and installation scripts
- Build and run Python-based ROS 2 packages
- Write launch files for Python nodes
- Test Python-based ROS 2 packages

## Related Content

For deeper understanding of related concepts, see:
- [Introduction to Physical AI](../week-1-2/introduction-to-physical-ai.md) - For foundational concepts
- [ROS 2 Architecture and Core Concepts](./ros2-architecture.md) - For understanding ROS 2 fundamentals
- [Nodes, topics, services, and actions](./ros2-nodes-topics.md) - For detailed communication patterns
- [Launch files and parameter management](./ros2-launch-files.md) - For advanced launch configuration
- [Gazebo simulation environment setup](../week-6-7/gazebo-setup.md) - For simulation integration
- [NVIDIA Isaac SDK and Isaac Sim](../week-8-10/isaac-sdk-sim.md) - For advanced ROS 2 integration
- [Humanoid robot kinematics and dynamics](../week-11-12/humanoid-kinematics-dynamics.md) - For robotics applications

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/ros2-examples/simple_publisher.py` - Basic publisher node example
- `docs/static/code-examples/ros2-examples/simple_subscriber.py` - Basic subscriber node example

## Exercises

1. Create a ROS 2 package with a Python node that publishes sensor data
2. Implement a service server in Python that processes incoming requests
3. Create a launch file that starts multiple Python nodes with parameters
4. Write unit tests for your Python ROS 2 nodes
5. Run and experiment with the provided ROS 2 examples to understand basic concepts