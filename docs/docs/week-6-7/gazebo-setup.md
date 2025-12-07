---
sidebar_position: 1
title: "Gazebo Simulation Environment Setup"
---

# Gazebo Simulation Environment Setup

Gazebo is a powerful 3D simulation environment for robotics that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. This section covers setting up and configuring Gazebo for robot simulation.

## Introduction to Gazebo

### What is Gazebo?
Gazebo is a 3D simulation environment that provides:
- **Realistic Physics**: Accurate simulation of rigid body dynamics, contact forces, and collisions
- **High-Quality Graphics**: Advanced rendering with support for shadows, textures, and lighting
- **Sensor Simulation**: Realistic simulation of cameras, LIDAR, IMUs, and other sensors
- **Programmatic Interfaces**: APIs for controlling simulation and accessing data
- **Large Model Database**: Extensive collection of robots, objects, and environments

### Key Features
- **Physics Engines**: Support for ODE, Bullet, and DART physics engines
- **Multi-Robot Simulation**: Simultaneous simulation of multiple robots
- **ROS Integration**: Native support for ROS and ROS 2 communication
- **Plugin Architecture**: Extensible through plugins for custom functionality
- **Cloud Simulation**: Support for cloud-based simulation environments

## Installing Gazebo

### System Requirements
- **Operating System**: Ubuntu 20.04/22.04 or equivalent Linux distribution
- **Graphics**: OpenGL 2.1+ compatible GPU with dedicated memory
- **RAM**: 8GB+ recommended for complex simulations
- **CPU**: Multi-core processor for physics simulation

### Installation Options
- **Gazebo Classic**: Legacy version with mature features
- **Gazebo Garden/Harmonic**: Newer versions with improved architecture
- **Ignition Gazebo**: Modular simulation framework (now part of Gazebo project)

### Installation Command
```bash
# For ROS 2 Humble/Foxy with Gazebo Classic
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins

# For standalone Gazebo Garden
sudo apt install gz-garden
```

## Basic Gazebo Concepts

### World Files
World files define the simulation environment:
- **Environment Description**: Lighting, models, physics properties
- **Terrain**: Ground planes, elevation maps, custom terrain
- **Weather**: Atmospheric conditions, wind simulation
- **Initial Conditions**: Starting positions and states

### Model Files
Models represent robots and objects in simulation:
- **Visual Elements**: Appearance, textures, colors
- **Collision Geometry**: Physics properties for collision detection
- **Inertial Properties**: Mass, center of mass, moments of inertia
- **Joints**: Connections between model parts

### Physics Properties
- **Gravity**: Gravitational acceleration settings
- **Damping**: Velocity and angular damping coefficients
- **Solver Parameters**: Physics solver configuration
- **Material Properties**: Friction, restitution coefficients

## Gazebo Interfaces

### GUI Interface
- **3D Viewport**: Interactive 3D visualization
- **Model Palette**: Library of available models
- **Timeline Control**: Play, pause, step through simulation
- **Properties Panel**: Adjust model and world properties

### Command Line Interface
```bash
# Launch Gazebo with a world file
gz sim -r my_world.sdf

# Launch with verbose output
gz sim -v 4 -r my_world.sdf

# Launch without GUI
gz sim -s my_world.sdf
```

### Programmatic Interface
- **Gazebo Transport**: Message passing system
- **REST API**: HTTP-based control interface
- **Python/C++ APIs**: Direct programmatic access

## ROS 2 Integration

### Gazebo ROS Packages
- **gazebo_ros**: Core ROS 2 integration package
- **gazebo_plugins**: Collection of ROS 2 plugins
- **gazebo_msgs**: ROS 2 message definitions for Gazebo

### Launching with ROS 2
```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch Gazebo with ROS 2 bridge
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                get_package_share_directory('my_robot_description'),
                'worlds',
                'my_world.sdf'
            ])
        }.items()
    )

    return LaunchDescription([
        gazebo,
        # Additional nodes
    ])
```

### Gazebo Plugins for ROS 2
- **Diff Drive Plugin**: Differential drive robot simulation
- **Joint State Publisher**: Publish joint states to ROS
- **Camera Plugin**: Camera sensor simulation with ROS 2 interface
- **IMU Plugin**: IMU sensor simulation
- **LIDAR Plugin**: 2D/3D LIDAR simulation

## Setting Up Robot Models

### URDF to SDF Conversion
```bash
# Convert URDF to SDF for Gazebo
gz sdf -p robot.urdf > robot.sdf

# Or use xacro to generate URDF with Gazebo extensions
xacro robot.xacro --inorder > robot.urdf
```

### Adding Gazebo-Specific Tags
```xml
<!-- In URDF/XACRO file -->
<gazebo>
  <material>Gazebo/Blue</material>
  <turnGravityOff>false</turnGravityOff>
</gazebo>

<!-- For links -->
<gazebo reference="link_name">
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<!-- For joints -->
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <ros>
      <namespace>/robot_name</namespace>
    </ros>
    <update_rate>30</update_rate>
    <joint_name>joint_name</joint_name>
  </plugin>
</gazebo>
```

## Simulation Control

### Starting and Stopping Simulation
```bash
# Start simulation with world
gz sim -r my_world.sdf

# Start without running simulation
gz sim my_world.sdf

# Control simulation via command line
gz service -s /world/default/control --reqtype ignition.msgs.WorldControl --reptype ignition.msgs.Boolean --timeout 5000 --req 'pause: true'
```

### Simulation Services
- **/world/[name]/control**: Pause/resume simulation
- **/world/[name]/reset**: Reset simulation state
- **/world/[name]/spawn**: Spawn new models
- **/world/[name]/delete**: Delete models

## Common Simulation Scenarios

### Single Robot Simulation
```xml
<!-- Simple world file -->
<sdf version="1.7">
  <world name="default">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Multi-Robot Simulation
```xml
<!-- Multi-robot world -->
<sdf version="1.7">
  <world name="multi_robot">
    <!-- Robot 1 -->
    <include>
      <uri>model://robot_1</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>

    <!-- Robot 2 -->
    <include>
      <uri>model://robot_2</uri>
      <pose>2 0 0.5 0 0 0</pose>
    </include>

    <!-- Environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

## Performance Optimization

### Physics Settings
- **Update Rate**: Balance accuracy vs. performance
- **Real-time Factor**: Control simulation speed relative to real time
- **Contact Coefficients**: Tune for stability and performance
- **Solver Iterations**: Adjust for convergence

### Graphics Settings
- **Rendering Quality**: Adjust based on hardware capabilities
- **LOD (Level of Detail)**: Simplify complex models at distance
- **Texture Resolution**: Balance visual quality with performance
- **Shadow Quality**: Reduce shadows for better performance

### Model Optimization
- **Collision Simplification**: Use simplified geometry for collision
- **Visual vs. Collision**: Separate visual and collision models
- **Joint Limits**: Properly constrain joint ranges
- **Inertial Properties**: Accurate mass and inertia values

## Troubleshooting Common Issues

### Physics Issues
- **Jittery Movement**: Check inertial properties and joint limits
- **Penetration**: Increase contact stiffness or reduce time step
- **Instability**: Reduce solver step size or increase iterations

### Graphics Issues
- **Missing Textures**: Verify model paths and file permissions
- **Poor Performance**: Reduce rendering quality or simplify models
- **Visual Artifacts**: Check for overlapping geometry

### ROS Integration Issues
- **No Communication**: Verify ROS 2 domain ID and network setup
- **Delayed Messages**: Check QoS settings and network configuration
- **Missing TF**: Ensure robot state publisher is running

## Learning Objectives

After completing this section, you should be able to:
- Install and configure Gazebo simulation environment
- Create and configure world files for simulation
- Integrate Gazebo with ROS 2 systems
- Set up robot models with appropriate Gazebo extensions
- Control simulation using command-line and programmatic interfaces
- Optimize simulation performance for complex scenarios

## Related Content

For deeper understanding of related concepts, see:
- [Introduction to Physical AI](../week-1-2/introduction-to-physical-ai.md) - For foundational concepts
- [ROS 2 Architecture and Core Concepts](../week-3-5/ros2-architecture.md) - For ROS 2 integration details
- [URDF and SDF robot description formats](./urdf-sdf-formats.md) - For detailed robot modeling
- [NVIDIA Isaac SDK and Isaac Sim](../week-8-10/isaac-sdk-sim.md) - For alternative simulation platforms
- [AI-powered perception and manipulation](../week-8-10/ai-perception-manipulation.md) - For advanced perception in simulation
- [Sim-to-real transfer techniques](../week-8-10/sim-to-real-transfer.md) - For bridging simulation and reality
- [Humanoid robot kinematics and dynamics](../week-11-12/humanoid-kinematics-dynamics.md) - For humanoid-specific simulation

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/ros2-examples/simple_publisher.py` - Example of ROS 2 node that can interface with Gazebo
- `docs/static/code-examples/ros2-examples/simple_subscriber.py` - Example of ROS 2 node that can interface with Gazebo

## Exercises

1. Install Gazebo and launch a simple world with a robot model
2. Create a custom world file with multiple objects and environmental features
3. Configure a robot model with Gazebo-specific tags for ROS 2 integration
4. Implement a launch file that starts Gazebo with your robot and ROS 2 bridge
5. Experiment with the provided ROS 2 examples in a Gazebo environment