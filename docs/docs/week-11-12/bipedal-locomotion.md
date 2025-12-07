---
sidebar_position: 2
title: "Bipedal Locomotion and Balance Control"
---

# Bipedal Locomotion and Balance Control

Bipedal locomotion is one of the most challenging aspects of humanoid robotics, requiring sophisticated control strategies to achieve stable and efficient walking patterns. This section covers the principles of bipedal gait generation, balance control, and stability maintenance.

## Fundamentals of Bipedal Locomotion

### Gait Phases

Bipedal walking consists of several distinct phases:

1. **Double Support Phase**: Both feet are in contact with the ground
2. **Single Support Phase**: Only one foot is in contact with the ground
3. **Swing Phase**: The non-support leg moves forward
4. **Contact Phase**: The swing foot makes contact with the ground

Understanding these phases is crucial for generating stable walking patterns.

### Walking Pattern Generation

#### Footstep Planning

Footstep planning determines where and when the feet should be placed:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class FootstepPlanner:
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05):
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height

    def plan_footsteps(self, start_pos, goal_pos, num_steps):
        """
        Plan a sequence of footsteps from start to goal
        """
        footsteps = []

        # Calculate direction vector
        direction = goal_pos[:2] - start_pos[:2]
        distance = np.linalg.norm(direction)

        if distance == 0:
            return footsteps

        direction = direction / distance  # Normalize

        # Generate footsteps
        current_pos = start_pos.copy()

        for i in range(num_steps):
            # Alternate between left and right foot
            if i % 2 == 0:  # Left foot
                lateral_offset = np.array([-self.step_width/2, 0])
            else:  # Right foot
                lateral_offset = np.array([self.step_width/2, 0])

            # Rotate lateral offset based on walking direction
            rotation_matrix = np.array([
                [direction[0], -direction[1]],
                [direction[1], direction[0]]
            ])
            offset = rotation_matrix @ lateral_offset

            # Calculate foot position
            foot_pos = current_pos[:2] + direction * self.step_length * (i + 1) + offset
            foot_yaw = np.arctan2(direction[1], direction[0])

            footsteps.append({
                'position': np.append(foot_pos, current_pos[2]),
                'yaw': foot_yaw,
                'step_number': i + 1,
                'support_leg': 'left' if i % 2 == 0 else 'right'
            })

        return footsteps
```

#### Trajectory Generation

Generating smooth trajectories for the feet and center of mass:

```python
def generate_foot_trajectory(foot_pos, lift_height=0.05, step_time=1.0, dt=0.01):
    """
    Generate a smooth trajectory for a foot movement
    """
    # Time parameters
    lift_time = step_time * 0.2  # 20% of time for lifting
    move_time = step_time * 0.6  # 60% of time for moving
    place_time = step_time * 0.2  # 20% of time for placing

    # Calculate trajectory points
    trajectory = []
    t = 0

    # Pre-lift phase (if starting from ground)
    start_pos = foot_pos - np.array([0, 0, lift_height/2])  # Start slightly raised

    while t <= step_time:
        if t < lift_time:
            # Lift phase - parabolic lift
            progress = t / lift_time
            height = start_pos[2] + lift_height * (1 - np.cos(np.pi * progress)) / 2
            pos = start_pos + np.array([0, 0, height - start_pos[2]])
        elif t < lift_time + move_time:
            # Move phase - linear movement with constant height
            progress = (t - lift_time) / move_time
            pos = start_pos + np.array([0, 0, lift_height]) + \
                  (foot_pos - start_pos - np.array([0, 0, lift_height])) * progress
        else:
            # Place phase - parabolic lowering
            progress = (t - lift_time - move_time) / place_time
            height = lift_height * (1 - np.cos(np.pi * progress)) / 2
            pos = foot_pos + np.array([0, 0, height])

        trajectory.append(pos.copy())
        t += dt

    return np.array(trajectory)
```

## Balance Control Strategies

### Center of Mass Control

Maintaining the center of mass within the support polygon is fundamental for balance:

```python
class BalanceController:
    def __init__(self, com_height=0.8, control_gain=10.0):
        self.com_height = com_height
        self.k = np.sqrt(9.81 / com_height)  # Pendulum frequency
        self.control_gain = control_gain

    def calculate_balance_correction(self, current_com, desired_com, com_velocity):
        """
        Calculate balance correction based on CoM deviation
        """
        # Calculate error
        com_error = desired_com - current_com

        # Apply control law (simplified)
        correction = self.control_gain * com_error - 2 * np.sqrt(self.control_gain) * com_velocity

        return correction
```

### Capture Point Control

The capture point is where the robot would need to step to come to a complete stop:

```python
def calculate_capture_point(com_position, com_velocity, gravity=9.81):
    """
    Calculate the capture point for stopping the robot
    """
    com_height = com_position[2]
    omega = np.sqrt(gravity / com_height)

    capture_point = com_position[:2] + com_velocity[:2] / omega

    return capture_point

def capture_point_controller(current_com, current_vel, target_capture_point, dt):
    """
    Control the robot to reach a target capture point
    """
    # Calculate current capture point
    current_capture = calculate_capture_point(current_com, current_vel)

    # Calculate required step location
    step_location = target_capture_point

    return step_location
```

### Linear Inverted Pendulum Mode (LIPM)

The LIPM is a common model for humanoid walking:

```python
class LIPMController:
    def __init__(self, com_height=0.8, dt=0.01):
        self.com_height = com_height
        self.omega = np.sqrt(9.81 / com_height)
        self.dt = dt

    def update(self, current_com, current_vel, zmp_reference):
        """
        Update CoM position and velocity based on ZMP reference
        """
        # LIPM dynamics: com'' = omega^2 * (com - zmp)
        com_acc = self.omega**2 * (current_com - zmp_reference)

        # Integrate to get new position and velocity
        new_vel = current_vel + com_acc * self.dt
        new_com = current_com + new_vel * self.dt

        return new_com, new_vel
```

## Walking Pattern Generators

### Preview Control

Preview control uses future reference information to generate stable walking patterns:

```python
class PreviewController:
    def __init__(self, preview_steps=20, dt=0.01):
        self.preview_steps = preview_steps
        self.dt = dt

    def generate_zmp_trajectory(self, start_pos, goal_pos, step_length=0.3):
        """
        Generate ZMP trajectory with preview control
        """
        # Simplified implementation - in practice, this would involve
        # solving Riccati equations for optimal control
        num_steps = int(np.linalg.norm(goal_pos - start_pos) / step_length)

        # Generate ZMP reference based on desired walking pattern
        zmp_trajectory = []
        for i in range(self.preview_steps):
            # Calculate ZMP position based on desired foot placement
            t = i * self.dt
            zmp_x = start_pos[0] + (goal_pos[0] - start_pos[0]) * (t / (self.preview_steps * self.dt))
            zmp_y = start_pos[1] + (goal_pos[1] - start_pos[1]) * (t / (self.preview_steps * self.dt))
            zmp_trajectory.append([zmp_x, zmp_y])

        return np.array(zmp_trajectory)
```

## Stability Analysis

### Zero Moment Point (ZMP) Stability

ZMP-based stability analysis is fundamental for bipedal robots:

```python
def check_zmp_stability(zmp_position, support_polygon):
    """
    Check if ZMP is within the support polygon
    """
    # Simplified check for rectangular support polygon
    min_x, max_x = support_polygon['x_range']
    min_y, max_y = support_polygon['y_range']

    return min_x <= zmp_position[0] <= max_x and min_y <= zmp_position[1] <= max_y

def calculate_support_polygon(foot_positions):
    """
    Calculate support polygon from foot positions
    """
    if len(foot_positions) == 1:
        # Single support - small polygon around foot
        foot = foot_positions[0]
        margin = 0.05  # 5cm margin
        return {
            'x_range': [foot[0] - margin, foot[0] + margin],
            'y_range': [foot[1] - margin, foot[1] + margin]
        }
    else:
        # Double support - polygon encompassing both feet
        all_x = [pos[0] for pos in foot_positions]
        all_y = [pos[1] for pos in foot_positions]
        return {
            'x_range': [min(all_x), max(all_x)],
            'y_range': [min(all_y), max(all_y)]
        }
```

## Advanced Control Techniques

### Model Predictive Control (MPC)

MPC is effective for humanoid balance control:

```python
def model_predictive_balance_control(current_state, reference_trajectory,
                                   prediction_horizon=10, dt=0.01):
    """
    Simplified MPC for balance control
    """
    # This would typically involve solving an optimization problem
    # For this example, we'll use a simplified approach

    # Predict future states based on current control
    predicted_states = []
    current = current_state.copy()

    for i in range(prediction_horizon):
        # Apply simple control law
        error = reference_trajectory[i] - current
        control_input = 10.0 * error  # Simple proportional control

        # Update state (simplified dynamics)
        current += control_input * dt
        predicted_states.append(current.copy())

    return np.array(predicted_states)
```

### Whole-Body Control

Integrating balance with full-body motion:

```python
class WholeBodyController:
    def __init__(self):
        self.balance_controller = BalanceController()
        self.ik_solver = InverseKinematicsSolver()  # Assumed implementation

    def compute_control(self, desired_com, desired_feet_pos, current_state):
        """
        Compute full-body control commands
        """
        # Balance control
        balance_correction = self.balance_controller.calculate_balance_correction(
            current_state['com'], desired_com, current_state['com_vel']
        )

        # Foot placement control
        desired_foot_positions = self.plan_foot_positions(
            desired_com, desired_feet_pos, current_state
        )

        # Inverse kinematics for full body
        joint_commands = self.ik_solver.solve(
            desired_com, desired_foot_positions, current_state
        )

        return joint_commands, balance_correction

    def plan_foot_positions(self, desired_com, desired_feet_pos, current_state):
        """
        Plan foot positions considering balance and desired motion
        """
        # Combine balance requirements with desired foot positions
        capture_point = calculate_capture_point(
            current_state['com'], current_state['com_vel']
        )

        # Adjust foot positions based on balance requirements
        adjusted_feet_pos = desired_feet_pos.copy()
        # Add balance correction logic here

        return adjusted_feet_pos
```

## Practical Implementation Considerations

### Sensor Integration

Balance control requires integration of multiple sensors:

- **IMU**: Provides orientation and angular velocity
- **Force/Torque sensors**: Measure ground reaction forces
- **Joint encoders**: Provide joint position feedback
- **Vision systems**: For external reference and obstacle detection

### Real-time Performance

Bipedal control requires real-time performance:

- Control loops typically run at 200-1000 Hz
- Prediction and planning must be computationally efficient
- Robust error handling for sensor failures

## Learning Objectives

After completing this section, you should be able to:
- Understand the phases of bipedal locomotion
- Implement footstep planning algorithms
- Design balance control systems using ZMP and capture point
- Apply advanced control techniques like MPC to humanoid robots
- Integrate sensor feedback for stable walking

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/capstone/humanoid_capstone_template.py` - Template for humanoid robot integration

## Exercises

1. Implement a footstep planner for a humanoid robot
2. Create a simple ZMP-based balance controller
3. Generate walking trajectories using the inverted pendulum model
4. Design a capture point controller for stopping the robot
5. Integrate sensor feedback into a balance control system