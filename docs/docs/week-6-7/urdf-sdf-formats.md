---
sidebar_position: 2
title: "URDF and SDF Robot Description Formats"
---

# URDF and SDF Robot Description Formats

This section covers the Unified Robot Description Format (URDF) and Simulation Description Format (SDF), which are used to describe robot models in ROS and Gazebo respectively.

## Introduction to Robot Description Formats

### Purpose
Robot description formats allow you to:
- Define robot kinematics and dynamics
- Specify visual and collision properties
- Describe sensors and actuators
- Configure simulation parameters
- Share robot models across different platforms

### URDF vs SDF
- **URDF (Unified Robot Description Format)**: Primarily used in ROS for robot description
- **SDF (Simulation Description Format)**: Used by Gazebo for simulation
- **Conversion**: URDF can be converted to SDF for simulation use

## URDF (Unified Robot Description Format)

### Basic Structure
```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

### Links
Links represent rigid bodies in the robot:
- **Visual**: How the link appears in simulation/visualization
- **Collision**: Geometry used for collision detection
- **Inertial**: Physical properties for dynamics simulation

#### Visual Elements
```xml
<link name="link_name">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Choose one geometry type -->
      <box size="1 1 1"/>
      <!-- <cylinder radius="0.5" length="1"/> -->
      <!-- <sphere radius="0.5"/> -->
      <!-- <mesh filename="package://my_robot/meshes/link.dae"/> -->
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>
</link>
```

#### Collision Elements
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="1 1 1"/>
  </geometry>
</collision>
```

#### Inertial Elements
```xml
<inertial>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <mass value="1.0"/>
  <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
</inertial>
```

### Joints
Joints connect links and define their relative motion:
- **Revolute**: Rotational joint with limits
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint
- **Fixed**: No relative motion
- **Floating**: 6-DOF motion
- **Planar**: Motion on a plane

#### Joint Types Example
```xml
<!-- Revolute Joint -->
<joint name="revolute_joint" type="revolute">
  <parent link="base_link"/>
  <child link="arm_link"/>
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>

<!-- Fixed Joint -->
<joint name="fixed_joint" type="fixed">
  <parent link="base_link"/>
  <child link="sensor_link"/>
  <origin xyz="0.1 0 0.2" rpy="0 0 0"/>
</joint>
```

### Transmissions
Transmissions define how joints connect to actuators:
```xml
<transmission name="transmission_joint1">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint1">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="motor1">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## SDF (Simulation Description Format)

### Basic Structure
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_robot">
    <!-- Links -->
    <link name="base_link">
      <pose>0 0 0.5 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.2</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.2</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>
      <inertial>
        <mass>10</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>1.0</iyy>
          <iyz>0.0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Joints -->
    <joint name="joint_name" type="revolute">
      <parent>base_link</parent>
      <child>child_link</child>
      <pose>0 0 0.5 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

### SDF Extensions for Gazebo
```xml
<sdf version="1.7">
  <model name="my_robot">
    <!-- Standard model definition -->

    <!-- Gazebo-specific extensions -->
    <gazebo reference="base_link">
      <material>Gazebo/Blue</material>
      <turnGravityOff>false</turnGravityOff>
    </gazebo>

    <gazebo>
      <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.3</wheel_separation>
        <wheel_diameter>0.15</wheel_diameter>
      </plugin>
    </gazebo>
  </model>
</sdf>
```

## Xacro (XML Macros)

### Purpose
Xacro allows you to:
- Define reusable macros
- Use mathematical expressions
- Include other xacro files
- Reduce duplication in robot descriptions

### Basic Xacro Example
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">

  <!-- Properties -->
  <xacro:property name="robot_radius" value="0.15" />
  <xacro:property name="robot_height" value="0.3" />
  <xacro:property name="PI" value="3.1415926535897931" />

  <!-- Macro for wheel -->
  <xacro:macro name="wheel" params="prefix parent reflect">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="0 ${reflect*0.1} 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.02"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.02"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:wheel prefix="front_left" parent="base_link" reflect="1"/>
  <xacro:wheel prefix="front_right" parent="base_link" reflect="-1"/>

</robot>
```

## Converting URDF to SDF

### Command Line Conversion
```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or using xacro to generate URDF first
xacro robot.xacro > robot.urdf
gz sdf -p robot.urdf > robot.sdf
```

### Automatic Conversion in Launch Files
```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Convert URDF to SDF automatically
    robot_description = Command([
        'xacro ',
        get_package_share_directory('my_robot_description'),
        '/urdf/my_robot.xacro'
    ])

    return LaunchDescription([
        # Launch Gazebo with robot description
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('gazebo_ros'),
                '/launch/spawn_entity.launch.py'
            ]),
            launch_arguments={
                'entity': 'my_robot',
                'spawn_x': '0',
                'spawn_y': '0',
                'spawn_z': '0.5',
                'robot_namespace': '',
                'robot_sdf': robot_description
            }.items()
        )
    ])
```

## Best Practices

### URDF Best Practices
- **Use Xacro**: Reduce duplication with macros and properties
- **Proper Inertial Values**: Accurate mass and inertia for stable simulation
- **Collision Simplification**: Use simpler geometry for collision than visual
- **Joint Limits**: Always specify appropriate limits
- **Consistent Units**: Use SI units throughout

### SDF Best Practices
- **Physics Tuning**: Adjust contact parameters for stable simulation
- **LOD Models**: Use different levels of detail as needed
- **Gazebo Plugins**: Include appropriate plugins for ROS integration
- **Material Properties**: Specify appropriate friction and restitution

### File Organization
```
my_robot_description/
├── urdf/
│   ├── my_robot.xacro
│   ├── materials.xacro
│   └── transmission.xacro
├── meshes/
│   ├── visual/
│   └── collision/
├── launch/
│   └── display.launch.py
├── CMakeLists.txt
└── package.xml
```

## Debugging Robot Descriptions

### URDF Validation
```bash
# Check URDF syntax
check_urdf robot.urdf

# Get URDF info
urdf_to_graphiz robot.urdf
```

### Common Issues
- **Missing Dependencies**: Check mesh file paths and permissions
- **Invalid Inertial Values**: Ensure positive masses and valid inertia matrices
- **Joint Limit Issues**: Verify joint limits and positions
- **Collision Issues**: Check for proper collision geometry

## Learning Objectives

After completing this section, you should be able to:
- Create robot descriptions using URDF and SDF formats
- Use Xacro macros to reduce duplication in robot descriptions
- Convert between URDF and SDF formats
- Configure robot models with proper visual, collision, and inertial properties
- Include Gazebo-specific extensions in robot descriptions
- Debug common robot description issues

## Related Content

For deeper understanding of related concepts, see:
- [Introduction to Physical AI](../week-1-2/introduction-to-physical-ai.md) - For foundational concepts
- [ROS 2 Architecture and Core Concepts](../week-3-5/ros2-architecture.md) - For ROS 2 integration
- [Gazebo simulation environment setup](./gazebo-setup.md) - For simulation context
- [NVIDIA Isaac SDK and Isaac Sim](../week-8-10/isaac-sdk-sim.md) - For alternative simulation platforms
- [Humanoid robot kinematics and dynamics](../week-11-12/humanoid-kinematics-dynamics.md) - For advanced kinematics
- [Bipedal locomotion and balance control](../week-11-12/bipedal-locomotion.md) - For humanoid-specific modeling
- [Manipulation and grasping with humanoid hands](../week-11-12/manipulation-grasping.md) - For manipulation modeling

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/ros2-examples/simple_publisher.py` - Example of ROS 2 node that can interface with robot models
- `docs/static/code-examples/ros2-examples/simple_subscriber.py` - Example of ROS 2 node that can interface with robot models

## Exercises

1. Create a simple robot model using URDF with at least 3 links and 2 joints
2. Use Xacro to create a parametric robot model that can be reused
3. Add Gazebo-specific extensions to your URDF model
4. Convert your URDF to SDF and verify the conversion
5. Experiment with the provided ROS 2 examples to understand how they interact with robot models