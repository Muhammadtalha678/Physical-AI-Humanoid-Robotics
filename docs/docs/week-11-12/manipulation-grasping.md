---
sidebar_position: 3
title: "Manipulation and Grasping with Humanoid Hands"
---

# Manipulation and Grasping with Humanoid Hands

Humanoid manipulation involves complex coordination of multiple degrees of freedom to perform dexterous tasks. This section covers the principles of robotic manipulation, grasping strategies, and control techniques specific to humanoid robots with anthropomorphic hands.

## Introduction to Humanoid Manipulation

### Degrees of Freedom and Workspace

Humanoid robots typically have 7-8 degrees of freedom per arm, providing redundant solutions for reaching tasks. The human hand has 20+ degrees of freedom, making dexterous manipulation particularly challenging.

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidArm:
    def __init__(self):
        # Typical humanoid arm with 7 DOF
        self.joint_limits = np.array([
            [-2.0, 2.0],   # Shoulder yaw
            [-1.5, 1.5],   # Shoulder pitch
            [-3.0, 1.0],   # Shoulder roll
            [-2.0, 2.0],   # Elbow pitch
            [-2.0, 2.0],   # Forearm yaw
            [-1.5, 2.0],   # Wrist pitch
            [-2.0, 2.0]    # Wrist roll
        ])

        self.link_lengths = [0.15, 0.3, 0.3, 0.05]  # Simplified arm segments

    def forward_kinematics(self, joint_angles):
        """
        Calculate forward kinematics for the arm
        """
        # Simplified implementation - in practice this would involve
        # complex transformation matrices
        position = np.zeros(3)
        rotation = R.from_euler('xyz', [0, 0, 0])

        # Calculate transformations for each joint
        for i, angle in enumerate(joint_angles):
            # This is a simplified representation
            # Real implementation would use DH parameters or other methods
            pass

        return position, rotation
```

### Anthropomorphic Hand Design

Humanoid hands are designed to mimic human hand capabilities:

- **Opposable thumb**: Enables precision grasps
- **Multiple fingers**: Provide multiple contact points
- **Tactile sensing**: Feedback for grasp stability
- **Underactuation**: Often use fewer actuators than DOF

## Grasp Planning and Analysis

### Grasp Types

Humanoid robots can perform various grasp types:

1. **Power Grasps**: For lifting and carrying heavy objects
2. **Precision Grasps**: For fine manipulation tasks
3. **Pinch Grasps**: Using thumb and one finger
4. **Lateral Grasps**: Using thumb and side of index finger

```python
class GraspPlanner:
    def __init__(self):
        self.grasp_types = {
            'cylindrical': self.plan_cylindrical_grasp,
            'spherical': self.plan_spherical_grasp,
            'lateral': self.plan_lateral_grasp,
            'tip': self.plan_tip_grasp
        }

    def plan_grasp(self, object_shape, object_size, grasp_type='cylindrical'):
        """
        Plan a grasp based on object properties and desired grasp type
        """
        if grasp_type in self.grasp_types:
            return self.grasp_types[grasp_type](object_shape, object_size)
        else:
            raise ValueError(f"Unknown grasp type: {grasp_type}")

    def plan_cylindrical_grasp(self, object_shape, object_size):
        """
        Plan a cylindrical grasp for a cylindrical object
        """
        # Find optimal grasp points around the cylinder
        grasp_points = []

        # Calculate grasp positions
        radius = object_size[0] / 2  # Assuming first dimension is diameter
        height = object_size[1]      # Assuming second dimension is height

        # Calculate grasp positions on opposite sides
        grasp_pos1 = np.array([0, radius, height/2])
        grasp_pos2 = np.array([0, -radius, height/2])

        # Calculate grasp orientations
        grasp_orient1 = R.from_euler('xyz', [0, 0, np.pi/2]).as_matrix()
        grasp_orient2 = R.from_euler('xyz', [0, 0, -np.pi/2]).as_matrix()

        return {
            'grasp_points': [grasp_pos1, grasp_pos2],
            'orientations': [grasp_orient1, grasp_orient2],
            'grasp_type': 'cylindrical'
        }
```

### Grasp Stability Analysis

Analyzing whether a grasp will be stable:

```python
def calculate_grasp_quality(grasp_points, object_com, contact_forces, friction_coeff=0.8):
    """
    Calculate grasp quality based on contact points and forces
    """
    # Calculate grasp matrix
    G = np.zeros((6, len(grasp_points) * 3))  # 6 DOF, 3 forces per contact

    for i, (point, force) in enumerate(zip(grasp_points, contact_forces)):
        # Position vector from object COM to contact point
        r = point - object_com

        # Grasp matrix row for this contact
        G[0:3, i*3:(i+1)*3] = np.eye(3)  # Forces
        G[3:6, i*3:(i+1)*3] = np.cross(r, np.eye(3))  # Torques

    # Calculate grasp quality metric (condition number of grasp matrix)
    U, s, Vt = np.linalg.svd(G)
    quality = np.min(s) / np.max(s) if np.max(s) > 0 else 0

    return quality

def check_grasp_stability(object_mass, grasp_points, contact_forces, gravity=9.81):
    """
    Check if the grasp can support the object against gravity
    """
    # Calculate total upward force
    total_upward_force = sum(f[2] for f in contact_forces)  # Assuming z is up

    # Required force to support object
    required_force = object_mass * gravity

    return total_upward_force >= required_force
```

## Dexterous Manipulation

### In-Hand Manipulation

Moving objects within the hand without releasing:

```python
class InHandManipulator:
    def __init__(self, hand_model):
        self.hand_model = hand_model

    def roll_object(self, object_pose, target_pose, finger_positions):
        """
        Roll an object from current pose to target pose
        """
        # Calculate required finger movements to roll object
        pose_diff = target_pose - object_pose
        finger_commands = self.calculate_finger_trajectories(
            pose_diff, finger_positions
        )

        return finger_commands

    def reposition_grasp(self, current_grasp, target_grasp):
        """
        Reposition the grasp without releasing the object
        """
        # Plan a sequence of finger movements to change grasp configuration
        movement_sequence = []

        # Calculate intermediate grasp configurations
        for i in range(10):  # 10 intermediate steps
            t = i / 10.0
            intermediate_grasp = self.interpolate_grasps(
                current_grasp, target_grasp, t
            )
            movement_sequence.append(intermediate_grasp)

        return movement_sequence

    def interpolate_grasps(self, grasp1, grasp2, t):
        """
        Interpolate between two grasp configurations
        """
        interpolated = {}
        for key in grasp1.keys():
            if isinstance(grasp1[key], np.ndarray):
                interpolated[key] = grasp1[key] * (1-t) + grasp2[key] * t
            else:
                interpolated[key] = grasp1[key]
        return interpolated
```

### Multi-Finger Coordination

Coordinating multiple fingers for complex tasks:

```python
class MultiFingerController:
    def __init__(self, num_fingers=5):
        self.num_fingers = num_fingers
        self.finger_controllers = [FingerController(i) for i in range(num_fingers)]

    def coordinate_fingers(self, task, object_properties):
        """
        Coordinate fingers for a specific manipulation task
        """
        # Determine finger roles based on task
        finger_roles = self.assign_finger_roles(task, object_properties)

        # Generate individual finger commands
        finger_commands = []
        for i, role in enumerate(finger_roles):
            command = self.finger_controllers[i].execute_role(role, object_properties)
            finger_commands.append(command)

        return finger_commands

    def assign_finger_roles(self, task, object_properties):
        """
        Assign roles to fingers based on task requirements
        """
        roles = ['thumb']  # Always assign thumb role first

        if task == 'cylindrical_grasp':
            roles.extend(['oppose', 'support', 'support', 'support'])
        elif task == 'tip_grasp':
            roles.extend(['index', 'middle', 'none', 'none'])
        elif task == 'large_object':
            roles.extend(['index', 'middle', 'ring', 'pinky'])
        else:
            roles.extend(['support'] * (self.num_fingers - 1))

        return roles
```

## Control Strategies for Manipulation

### Impedance Control

Impedance control allows for compliant manipulation:

```python
class ImpedanceController:
    def __init__(self, stiffness=1000, damping=200, mass=1):
        self.stiffness = stiffness
        self.damping = damping
        self.mass = mass

    def calculate_impedance_force(self, position_error, velocity_error):
        """
        Calculate force based on impedance model
        """
        spring_force = self.stiffness * position_error
        damper_force = self.damping * velocity_error
        total_force = spring_force + damper_force

        return total_force

    def adapt_impedance(self, task_phase):
        """
        Adapt impedance based on task requirements
        """
        if task_phase == 'approach':
            # Low stiffness for safe approach
            self.stiffness = 100
        elif task_phase == 'grasp':
            # High stiffness for firm grasp
            self.stiffness = 2000
        elif task_phase == 'manipulate':
            # Variable stiffness based on task
            self.stiffness = 1000
        elif task_phase == 'release':
            # Low stiffness to avoid damage
            self.stiffness = 100
```

### Force Control

Force control is essential for safe manipulation:

```python
class ForceController:
    def __init__(self, desired_force, force_tolerance=5.0):
        self.desired_force = desired_force
        self.force_tolerance = force_tolerance
        self.integral_error = 0
        self.previous_error = 0

    def update_force_control(self, measured_force, dt):
        """
        Update force control based on measured force
        """
        error = self.desired_force - measured_force
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt if dt > 0 else 0

        # PID control
        kp, ki, kd = 1.0, 0.1, 0.05
        control_output = kp * error + ki * self.integral_error + kd * derivative_error

        self.previous_error = error

        return control_output

    def adjust_contact_force(self, desired_contact_force, current_contact_force):
        """
        Adjust contact force to desired level
        """
        force_error = desired_contact_force - current_contact_force

        if abs(force_error) > self.force_tolerance:
            # Adjust position to achieve desired force
            position_adjustment = 0.001 * np.sign(force_error)  # 1mm per 1N error
            return position_adjustment
        else:
            return 0.0  # No adjustment needed
```

## Tactile Sensing and Feedback

### Tactile Sensor Integration

Tactile feedback is crucial for dexterous manipulation:

```python
class TactileFeedbackSystem:
    def __init__(self, num_sensors=20):
        self.num_sensors = num_sensors
        self.sensor_threshold = 0.1  # Minimum pressure to register contact
        self.pressure_map = np.zeros(num_sensors)

    def process_tactile_data(self, raw_sensor_data):
        """
        Process raw tactile sensor data
        """
        # Convert raw data to pressure values
        pressure_values = self.convert_raw_to_pressure(raw_sensor_data)

        # Update pressure map
        self.pressure_map = pressure_values

        # Detect contact events
        contact_events = self.detect_contacts(pressure_values)

        return pressure_values, contact_events

    def detect_slip(self, current_pressures, previous_pressures, time_diff):
        """
        Detect potential slip based on pressure changes
        """
        pressure_changes = current_pressures - previous_pressures

        # High pressure change rate may indicate slip
        slip_indicators = np.abs(pressure_changes) / time_diff

        return slip_indicators > 100  # Threshold for slip detection

    def adjust_grasp(self, slip_detected):
        """
        Adjust grasp if slip is detected
        """
        if np.any(slip_detected):
            # Increase grasp force
            force_increase = 5.0  # 5N increase
            return force_increase
        else:
            return 0.0
```

## Learning-Based Manipulation

### Imitation Learning for Grasping

Learning from human demonstrations:

```python
class ImitationLearningGrasper:
    def __init__(self):
        self.demonstration_database = []
        self.feature_extractor = FeatureExtractor()
        self.policy_network = PolicyNetwork()

    def add_demonstration(self, state_sequence, action_sequence):
        """
        Add a successful grasp demonstration to the database
        """
        demonstration = {
            'states': state_sequence,
            'actions': action_sequence,
            'success': True
        }
        self.demonstration_database.append(demonstration)

    def learn_grasp_policy(self):
        """
        Learn a grasp policy from demonstrations
        """
        # Extract features from demonstrations
        states = []
        actions = []

        for demo in self.demonstration_database:
            for state, action in zip(demo['states'], demo['actions']):
                states.append(self.feature_extractor.extract(state))
                actions.append(action)

        # Train policy network
        self.policy_network.train(np.array(states), np.array(actions))

    def execute_grasp(self, current_state):
        """
        Execute grasp based on learned policy
        """
        features = self.feature_extractor.extract(current_state)
        action = self.policy_network.predict(features)
        return action
```

### Reinforcement Learning for Manipulation

```python
class RLManipulator:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer()

    def calculate_reward(self, state, action, next_state):
        """
        Calculate reward for manipulation task
        """
        reward = 0

        # Grasp stability reward
        if self.is_stable_grasp(next_state):
            reward += 10

        # Object position reward (closer to target)
        obj_pos = next_state['object_position']
        target_pos = next_state['target_position']
        dist_to_target = np.linalg.norm(obj_pos - target_pos)
        reward -= dist_to_target  # Negative distance reward

        # Penalty for excessive force
        if self.is_excessive_force(next_state):
            reward -= 5

        return reward

    def is_stable_grasp(self, state):
        """
        Check if the current grasp is stable
        """
        # Check if object is still in hand
        if not state['object_in_hand']:
            return False

        # Check if grasp forces are within reasonable bounds
        grasp_forces = state['grasp_forces']
        if np.any(grasp_forces > 50):  # 50N max per finger
            return False

        return True
```

## Humanoid-Specific Challenges

### Whole-Body Coordination

Humanoid manipulation requires coordination of the entire body:

```python
class WholeBodyManipulator:
    def __init__(self):
        self.arm_controller = ArmController()
        self.balance_controller = BalanceController()
        self.gaze_controller = GazeController()

    def coordinated_manipulation(self, task, object_pose, robot_state):
        """
        Perform manipulation with whole-body coordination
        """
        # Plan manipulation trajectory
        manipulation_plan = self.plan_manipulation(task, object_pose)

        # Adjust balance for manipulation
        balance_adjustment = self.calculate_balance_adjustment(
            manipulation_plan, robot_state
        )

        # Control gaze to track object
        gaze_command = self.calculate_gaze_command(object_pose)

        # Execute coordinated movement
        arm_command = self.arm_controller.execute_plan(manipulation_plan)

        return arm_command, balance_adjustment, gaze_command

    def calculate_balance_adjustment(self, manipulation_plan, robot_state):
        """
        Calculate balance adjustments needed for manipulation
        """
        # Predict center of mass shift due to manipulation
        predicted_com_shift = self.predict_com_shift(manipulation_plan)

        # Calculate required balance adjustments
        balance_command = self.balance_controller.calculate_balance_correction(
            robot_state['com'] + predicted_com_shift,
            robot_state['com_reference'],
            robot_state['com_velocity']
        )

        return balance_command
```

### Bilateral Manipulation

Using both arms for complex tasks:

```python
class BilateralManipulator:
    def __init__(self):
        self.left_arm = ArmController('left')
        self.right_arm = ArmController('right')
        self.coordinator = CoordinationController()

    def bimanual_task(self, task_description, object_poses):
        """
        Execute bimanual manipulation task
        """
        # Plan left and right arm trajectories
        left_plan = self.plan_left_arm_task(task_description, object_poses)
        right_plan = self.plan_right_arm_task(task_description, object_poses)

        # Coordinate both arms
        coordinated_plan = self.coordinator.coordinate_plans(left_plan, right_plan)

        # Execute in coordination
        left_command = self.left_arm.execute_plan(coordinated_plan['left'])
        right_command = self.right_arm.execute_plan(coordinated_plan['right'])

        return left_command, right_command

    def plan_left_arm_task(self, task, objects):
        """
        Plan left arm task based on overall task description
        """
        if task == 'pass_object':
            # Left arm receives object
            return self.plan_reach_and_grasp(objects['object_to_receive'])
        elif task == 'assemble':
            # Left arm holds base object
            return self.plan_stabilize_object(objects['base_object'])
        else:
            return self.plan_default_task()

    def plan_right_arm_task(self, task, objects):
        """
        Plan right arm task based on overall task description
        """
        if task == 'pass_object':
            # Right arm passes object
            return self.plan_release_object(objects['object_to_pass'])
        elif task == 'assemble':
            # Right arm manipulates parts
            return self.plan_assemble_parts(objects['parts'])
        else:
            return self.plan_default_task()
```

## Learning Objectives

After completing this section, you should be able to:
- Understand the kinematics and dynamics of humanoid manipulation
- Plan stable grasps for various object types
- Implement impedance and force control for safe manipulation
- Integrate tactile sensing for dexterous manipulation
- Coordinate whole-body motion for manipulation tasks
- Apply learning techniques to improve manipulation skills

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/capstone/humanoid_capstone_template.py` - Template for humanoid robot integration

## Exercises

1. Implement a grasp planner for cylindrical objects
2. Create an impedance controller for safe manipulation
3. Design a tactile feedback system for grasp stability
4. Implement a bimanual coordination controller
5. Develop a whole-body manipulation system for the humanoid