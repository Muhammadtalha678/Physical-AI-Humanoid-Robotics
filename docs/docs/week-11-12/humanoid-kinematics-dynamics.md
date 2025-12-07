---
sidebar_position: 1
title: "Humanoid Robot Kinematics and Dynamics"
---

# Humanoid Robot Kinematics and Dynamics

This section covers the mathematical foundations and physical principles governing the movement and forces in humanoid robots. Understanding kinematics and dynamics is essential for controlling the complex multi-link structures of humanoid robots.

## Introduction to Humanoid Kinematics

### Forward Kinematics

Forward kinematics determines the position and orientation of the end-effector given the joint angles. For humanoid robots, this involves complex kinematic chains with multiple degrees of freedom.

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def forward_kinematics(joint_angles, link_lengths):
    """
    Calculate forward kinematics for a simplified humanoid arm
    """
    # Base position
    position = np.array([0.0, 0.0, 0.0])

    # Calculate transformation matrices for each joint
    for i, angle in enumerate(joint_angles):
        # Rotation matrix for this joint
        rotation = R.from_euler('z', angle).as_matrix()

        # Translation along the link
        translation = np.array([link_lengths[i], 0, 0])

        # Apply transformation
        position = rotation @ position + translation

    return position
```

### Inverse Kinematics

Inverse kinematics solves for the joint angles required to achieve a desired end-effector position and orientation. This is particularly challenging for humanoid robots due to redundancy and multiple solutions.

```python
def inverse_kinematics_2d(target_pos, l1, l2):
    """
    Solve inverse kinematics for a 2D planar arm
    """
    x, y = target_pos

    # Distance from base to target
    r = np.sqrt(x**2 + y**2)

    # Check if target is reachable
    if r > l1 + l2:
        return None  # Target out of reach

    if r < abs(l1 - l2):
        return None  # Target too close

    # Calculate joint angles
    cos_theta2 = (l1**2 + l2**2 - r**2) / (2*l1*l2)
    theta2 = np.arccos(np.clip(cos_theta2, -1, 1))

    k1 = l1 + l2 * np.cos(theta2)
    k2 = l2 * np.sin(theta2)

    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return np.array([theta1, theta2])
```

### Kinematic Chains in Humanoid Robots

Humanoid robots have multiple kinematic chains:
- **Leg chains**: From hip to foot for each leg
- **Arm chains**: From shoulder to hand for each arm
- **Spine chain**: Connecting the torso segments
- **Head chain**: From neck to head

Each chain has specific constraints and degrees of freedom that must be considered during motion planning.

## Humanoid Dynamics

### Rigid Body Dynamics

The dynamics of humanoid robots are governed by the Newton-Euler or Lagrangian equations. For complex multi-link systems, the equations become:

```
M(q)q'' + C(q,q')q' + G(q) = τ
```

Where:
- M(q) is the mass matrix
- C(q,q') contains Coriolis and centrifugal terms
- G(q) is the gravity vector
- τ is the joint torque vector

### Center of Mass and Stability

Maintaining balance is critical for humanoid robots. The center of mass (CoM) must be kept within the support polygon defined by the feet.

```python
def calculate_com(robot_state, link_masses, link_positions):
    """
    Calculate center of mass for the robot
    """
    total_mass = sum(link_masses)
    com = np.zeros(3)

    for mass, pos in zip(link_masses, link_positions):
        com += mass * pos

    return com / total_mass

def is_stable(com_position, foot_positions):
    """
    Check if the robot is stable based on CoM position
    """
    # Calculate support polygon (simplified to bounding box)
    min_x = min(foot_positions[:, 0])
    max_x = max(foot_positions[:, 0])
    min_y = min(foot_positions[:, 1])
    max_y = max(foot_positions[:, 1])

    # Check if CoM is within support polygon
    return min_x <= com_position[0] <= max_x and min_y <= com_position[1] <= max_y
```

### Zero Moment Point (ZMP)

The ZMP is a critical concept for humanoid balance:

```python
def calculate_zmp(com_position, com_acceleration, gravity=9.81):
    """
    Calculate Zero Moment Point
    """
    x_com, y_com, z_com = com_position
    x_acc, y_acc, z_acc = com_acceleration

    zmp_x = x_com - (z_com - 0.0) * x_acc / gravity  # Assuming foot height is 0
    zmp_y = y_com - (z_com - 0.0) * y_acc / gravity

    return np.array([zmp_x, zmp_y])
```

## Walking Patterns and Gait Generation

### Inverted Pendulum Model

The linear inverted pendulum model (LIPM) is commonly used for humanoid walking:

```python
def lipm_walking_pattern(com_height, omega, initial_pos, target_pos, dt):
    """
    Generate walking pattern using Linear Inverted Pendulum Model
    """
    z_com = com_height
    k = omega  # sqrt(g / z_com)

    # Calculate ZMP trajectory
    zmp_trajectory = []
    com_trajectory = []

    # Simple implementation for single step
    current_pos = np.array(initial_pos)
    step_size = 0.3  # 30cm step
    step_duration = 1.0  # 1 second per step

    for t in np.arange(0, step_duration, dt):
        # ZMP moves to next foot position
        zmp_pos = current_pos + (target_pos - current_pos) * (t / step_duration)
        zmp_trajectory.append(zmp_pos)

        # Calculate CoM position based on ZMP
        com_x = zmp_pos[0] + (z_com / 9.81) * (k**2 * (current_pos[0] - zmp_pos[0]))
        com_y = zmp_pos[1] + (z_com / 9.81) * (k**2 * (current_pos[1] - zmp_pos[1]))

        com_trajectory.append([com_x, com_y, z_com])

    return np.array(zmp_trajectory), np.array(com_trajectory)
```

## Control Strategies

### Operational Space Control

Operational space control allows for direct control of end-effector forces and positions:

```python
def operational_space_control(robot_state, desired_pose, jacobian, mass_matrix):
    """
    Implement operational space control
    """
    # Calculate operational space mass matrix
    lambda_op = np.linalg.inv(jacobian @ np.linalg.inv(mass_matrix) @ jacobian.T)

    # Calculate desired operational space acceleration
    pose_error = desired_pose - current_pose  # Simplified
    desired_acc = kp * pose_error + kd * velocity_error  # Simplified

    # Calculate joint torques
    tau = jacobian.T @ (lambda_op @ desired_acc) + gravity_compensation

    return tau
```

## Humanoid-Specific Considerations

### Underactuation and Balance

Humanoid robots are typically underactuated systems where the number of actuators is less than the degrees of freedom in the system. This requires special control strategies:

- **Passive dynamics**: Using the natural dynamics of the system
- **Limit cycle control**: Creating stable periodic motions
- **Feedback linearization**: Linearizing the nonlinear dynamics

### Redundancy Resolution

With multiple degrees of freedom, humanoid robots have redundant solutions for reaching the same configuration. Common approaches include:

- **Null space projection**: Optimizing secondary objectives while achieving primary tasks
- **Pseudoinverse methods**: Finding minimum norm solutions
- **Task prioritization**: Handling multiple simultaneous tasks

## Learning Objectives

After completing this section, you should be able to:
- Understand forward and inverse kinematics for humanoid robots
- Apply dynamics principles to analyze humanoid robot motion
- Calculate center of mass and stability metrics
- Implement basic walking pattern generation algorithms
- Design operational space controllers for humanoid robots

## Related Content

For deeper understanding of related concepts, see:
- [Introduction to Physical AI](../week-1-2/introduction-to-physical-ai.md) - For foundational concepts
- [ROS 2 Architecture and Core Concepts](../week-3-5/ros2-architecture.md) - For robot software frameworks
- [Gazebo simulation environment setup](../week-6-7/gazebo-setup.md) - For simulating humanoid robots
- [NVIDIA Isaac SDK and Isaac Sim](../week-8-10/isaac-sdk-sim.md) - For advanced simulation platforms
- [Bipedal locomotion and balance control](./bipedal-locomotion.md) - For detailed walking control
- [Manipulation and grasping with humanoid hands](./manipulation-grasping.md) - For manipulation kinematics
- [Natural human-robot interaction design](./human-robot-interaction.md) - For interaction aspects
- [Capstone project requirements](../capstone-project.md) - For implementing complete humanoid systems

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/capstone/humanoid_capstone_template.py` - Template for humanoid robot integration

## Exercises

1. Implement forward kinematics for a simplified humanoid arm with 6 DOF
2. Solve inverse kinematics for a target position using numerical methods
3. Calculate the center of mass for a simple humanoid model
4. Generate a walking pattern using the inverted pendulum model
5. Create a simulation of a humanoid robot maintaining balance