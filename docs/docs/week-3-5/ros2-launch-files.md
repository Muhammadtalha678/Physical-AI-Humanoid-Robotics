---
sidebar_position: 4
title: "Launch Files and Parameter Management"
---

# Launch Files and Parameter Management

This section covers ROS 2 launch files and parameter management, which are essential for configuring and starting complex robot systems with multiple nodes.

## Launch Files Overview

### Purpose of Launch Files
Launch files allow you to:
- Start multiple nodes with a single command
- Configure nodes with parameters
- Set up remappings between topics/services
- Manage complex system initialization
- Provide reusable system configurations

### Launch File Types
- **XML-based**: Simple, declarative launch configuration
- **Python-based**: More flexible, programmable launch configuration
- **YAML-based**: Parameter configuration files

## Python Launch Files

### Basic Python Launch File Structure
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Define nodes here
        Node(
            package='my_package',
            executable='my_node',
            name='my_node_name',
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

### Advanced Launch File Features

#### Conditional Launch
```python
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    return LaunchDescription([
        declare_use_sim_time,
        Node(
            package='my_package',
            executable='my_node',
            name='my_node',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        )
    ])
```

#### Grouping Nodes
```python
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace

# Group nodes under a namespace
group_action = GroupAction(
    actions=[
        PushRosNamespace('robot1'),
        Node(
            package='navigation',
            executable='nav_node',
            name='nav_node'
        ),
        Node(
            package='localization',
            executable='loc_node',
            name='loc_node'
        )
    ]
)
```

#### Including Other Launch Files
```python
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Include another launch file
    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nav2_bringup'),
            '/launch/navigation_launch.py'
        ])
    )

    return LaunchDescription([
        nav_launch,
        # Additional nodes
    ])
```

## XML Launch Files

### Basic XML Structure
```xml
<launch>
  <node pkg="my_package" exec="my_node" name="my_node_name">
    <param name="param1" value="value1"/>
    <remap from="original_topic" to="new_topic"/>
  </node>
</launch>
```

### XML with Arguments
```xml
<launch>
  <arg name="robot_name" default="robot1"/>

  <node pkg="my_package" exec="my_node" name="$(var robot_name)_node">
    <param name="robot_name" value="$(var robot_name)"/>
  </node>
</launch>
```

## Parameter Management

### Parameter Sources
- **Launch files**: Parameters specified in launch files
- **YAML files**: External parameter configuration files
- **Command line**: Parameters passed at runtime
- **Parameter servers**: Dynamic parameter management

### YAML Parameter Files
```yaml
# robot_params.yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false
    global_frame: "map"
    robot_base_frame: "base_link"

my_robot_controller:  # Applies to specific node
  ros__parameters:
    cmd_vel_topic: "cmd_vel"
    odom_topic: "odom"
    max_velocity: 1.0
    min_velocity: -1.0
```

### Loading Parameters from YAML
```python
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get path to parameter file
    params_file = os.path.join(
        get_package_share_directory('my_package'),
        'config',
        'robot_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_package',
            executable='my_node',
            name='my_node',
            parameters=[params_file]
        )
    ])
```

## Parameter Server

### Parameter Server Architecture
The parameter server in ROS 2 is distributed:
- Each node maintains its own parameter service
- Parameters can be set/get at runtime
- Parameter callbacks allow nodes to respond to changes

### Parameter Service Usage
```bash
# Get parameter
ros2 param get /node_name parameter_name

# Set parameter
ros2 param set /node_name parameter_name value

# List parameters
ros2 param list /node_name

# Dump all parameters to file
ros2 param dump /node_name > params.yaml
```

### Parameter Callbacks in Nodes
```python
from rclpy.parameter import Parameter

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # Declare parameters with defaults
        self.declare_parameter('my_param', 'default_value')

        # Register parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'my_param' and param.type_ == Parameter.Type.STRING:
                self.get_logger().info(f'Parameter updated: {param.value}')
                # Handle parameter change
                return SetParametersResult(successful=True)
        return SetParametersResult(successful=True)
```

## Advanced Launch Features

### Event Handlers
```python
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.actions import RegisterEventHandler

# Execute action when node starts
on_node_start = RegisterEventHandler(
    OnProcessStart(
        target_action=node_action,
        on_start=[
            LogInfo(msg="Node started successfully")
        ]
    )
)
```

### Timed Actions
```python
from launch.actions import TimerAction

# Start node after delay
delayed_node = TimerAction(
    period=5.0,  # Wait 5 seconds
    actions=[node_action]
)
```

## Launch File Best Practices

### Organization
- Use descriptive names for launch files
- Group related functionality in separate launch files
- Use arguments to make launch files flexible
- Document launch file purpose and arguments

### Parameter Management
- Separate parameters into logical groups
- Use YAML files for complex parameter configurations
- Provide reasonable defaults
- Use consistent naming conventions

### Error Handling
- Validate launch file arguments
- Handle missing dependencies gracefully
- Provide informative error messages
- Test launch files in isolation

## Debugging Launch Files

### Common Issues
- **Node not found**: Check package installation and executable names
- **Parameter not loaded**: Verify YAML syntax and file paths
- **Permission errors**: Check file permissions and ownership
- **Dependency issues**: Ensure all required packages are installed

### Debugging Commands
```bash
# Check launch file syntax
ros2 launch --dry-run my_package my_launch.py

# Verbose output
ros2 launch my_package my_launch.py --log-level debug

# List all launch arguments
ros2 launch my_package my_launch.py --show-args
```

## Learning Objectives

After completing this section, you should be able to:
- Create and configure Python and XML launch files
- Manage parameters using YAML files and launch files
- Use launch arguments for flexible configurations
- Implement parameter callbacks in nodes
- Debug common launch file issues
- Organize complex robot systems using launch files

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/ros2-examples/simple_publisher.py` - Example node that can be launched with launch files
- `docs/static/code-examples/ros2-examples/simple_subscriber.py` - Example node that can be launched with launch files

## Exercises

1. Create a launch file that starts multiple nodes with different parameters
2. Implement a parameter server in a node with callback functionality
3. Create a YAML parameter file and load it in a launch file
4. Design a launch file that includes other launch files for modular configuration
5. Experiment with launch files using the provided ROS 2 examples