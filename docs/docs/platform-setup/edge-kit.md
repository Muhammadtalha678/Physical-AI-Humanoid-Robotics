---
sidebar_position: 7
title: "Physical AI Edge Kit Setup"
---

# Physical AI Edge Kit Setup

This guide provides instructions for setting up the Physical AI Edge Kit for humanoid robotics development. The Edge Kit provides a compact, powerful platform for developing and testing humanoid robotics applications with real hardware.

## Kit Components

### Hardware Components
- **Edge Computer**: NVIDIA Jetson Orin AGX (64GB) or equivalent
- **Robot Platform**: Humanoid robot with 16+ DOF, RGB-D camera, IMU
- **Power System**: Intelligent battery management with 2+ hours runtime
- **Communication**: WiFi 6, Bluetooth 5.2, Ethernet
- **Sensors**: RGB-D camera, IMU, force/torque sensors, tactile sensors
- **Accessories**: Charging station, calibration tools, safety equipment

### System Specifications
- **Processor**: NVIDIA ARM CPU + GPU
- **Memory**: 64GB LPDDR5
- **Storage**: 2TB NVMe SSD
- **Connectivity**: WiFi 6, Bluetooth 5.2, Gigabit Ethernet
- **Power**: 19V/150W input, intelligent battery management
- **Operating System**: Ubuntu 22.04 LTS with real-time kernel

## Initial Setup

### 1. Unboxing and Physical Setup
1. Carefully remove all components from packaging
2. Connect the robot to the edge computer using provided cables
3. Ensure all safety locks are secure before proceeding
4. Connect power supply to the edge computer
5. Allow 2-3 minutes for initial boot sequence

### 2. Network Configuration
```bash
# Connect to the robot's network or configure WiFi
# Check network status
ip addr show

# If using WiFi, configure connection
nmcli device wifi list
nmcli device wifi connect "RobotNetwork" password "password"
```

### 3. System Update and Configuration
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install real-time kernel (if not already installed)
sudo apt install linux-image-rt-generic

# Configure system for robotics applications
echo "kernel.sched_rt_runtime_us = -1" | sudo tee -a /etc/security/limits.conf
echo "kernel.sched_rt_period_us = 1000000" | sudo tee -a /etc/security/limits.conf
```

## Software Installation

### 1. ROS 2 Humble Hawksbill
```bash
# Set locale for UTF-8
locale  # check for UTF-8
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS 2 repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt upgrade -y

# Install ROS 2 packages
sudo apt install ros-humble-desktop ros-humble-ros-base
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install python3-colcon-common-extensions

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Robot-Specific Drivers and Packages
```bash
# Create workspace for robot packages
mkdir -p ~/robot_ws/src
cd ~/robot_ws/src

# Clone robot-specific packages
git clone https://github.com/physical-ai/edge-kit-ros2.git
git clone https://github.com/physical-ai/humanoid-drivers.git

# Build the workspace
cd ~/robot_ws
colcon build --symlink-install

# Source the workspace
echo "source ~/robot_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 3. Perception and AI Libraries
```bash
# Install Python dependencies for perception
pip3 install --user numpy scipy matplotlib
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install --user opencv-python open3d
pip3 install --user scikit-learn

# Install computer vision libraries
sudo apt install libopencv-dev python3-opencv
sudo apt install libpcl-dev pcl-tools
```

### 4. Simulation Environment (Optional but Recommended)
```bash
# Install Gazebo for simulation development
sudo apt install gz-harmonic

# Install Isaac Sim for advanced simulation (if supported)
# Follow NVIDIA Isaac Sim installation guide for ARM64
```

## Robot Calibration

### 1. Camera Calibration
```bash
# Launch camera calibration
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 image:=/camera/rgb/image_raw camera:=/camera/rgb

# Follow the calibration process and save results
# Calibration files will be saved to ~/.ros/camera_info/
```

### 2. IMU Calibration
```bash
# Launch IMU calibration
ros2 launch robot_bringup imu_calibration.launch.py

# Follow the calibration process for magnetometer, accelerometer, and gyroscope
```

### 3. Kinematic Calibration
```bash
# Launch kinematic calibration
ros2 launch robot_bringup kinematic_calibration.launch.py

# Use calibration target to calibrate DH parameters
# This ensures accurate forward and inverse kinematics
```

## Safety Configuration

### 1. Emergency Stop Setup
```bash
# Configure emergency stop behavior
echo "Configuring emergency stop system..."

# Create emergency stop node
ros2 run robot_safety emergency_stop_node &
```

### 2. Safety Parameters
```bash
# Set joint limits and safety parameters
ros2 param set /joint_state_broadcaster joint_limits "{
  'head_yaw': {'has_position_limits': True, 'min_position': -1.57, 'max_position': 1.57},
  'head_pitch': {'has_position_limits': True, 'min_position': -0.78, 'max_position': 0.78},
  'left_shoulder_pitch': {'has_position_limits': True, 'min_position': -1.57, 'max_position': 1.57},
  'left_shoulder_roll': {'has_position_limits': True, 'min_position': -2.0, 'max_position': 0.5},
  # Add all other joints with appropriate limits
}"
```

### 3. Collision Avoidance
```bash
# Configure self-collision avoidance
ros2 run robot_safety self_collision_avoidance &
```

## Network and Remote Access

### 1. SSH Configuration
```bash
# Enable SSH server
sudo systemctl enable ssh
sudo systemctl start ssh

# Generate SSH keys for secure access
ssh-keygen -t rsa -b 4096 -C "edge-kit@physical-ai"

# Configure SSH for robotics development
sudo nano /etc/ssh/sshd_config
# Ensure: PermitRootLogin no, PasswordAuthentication yes (for initial setup)
```

### 2. ROS 2 Network Configuration
```bash
# Configure ROS 2 for network communication
echo "export ROS_DOMAIN_ID=1" >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=0" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp" >> ~/.bashrc

# Configure DDS settings for real-time communication
echo "export CYCLONEDX_ENV_SECURITY=1" >> ~/.bashrc
```

## Performance Optimization

### 1. NVIDIA Jetson Configuration
```bash
# Install Jetson Performance tools
sudo apt install nvidia-jetson-performance

# Configure for maximum performance
sudo nvpmodel -m 0  # Maximum performance mode
sudo jetson_clocks  # Lock clocks at maximum

# Monitor performance
sudo tegrastats  # In another terminal
```

### 2. Real-time Configuration
```bash
# Configure real-time scheduling
echo "robot-kit soft rtprio 99" | sudo tee -a /etc/security/limits.conf
echo "robot-kit hard rtprio 99" | sudo tee -a /etc/security/limits.conf

# Configure CPU governor for performance
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl enable cpufrequtils
sudo systemctl start cpufrequtils
```

## Testing the Setup

### 1. Basic System Check
```bash
# Check system status
nvidia-smi  # Should show GPU utilization
tegrastats  # Should show system stats
free -h     # Check memory usage
df -h       # Check disk usage
```

### 2. ROS 2 Test
```bash
# Test ROS 2 installation
ros2 topic list
ros2 node list

# Test basic publisher/subscriber
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_cpp listener
```

### 3. Robot Hardware Test
```bash
# Launch robot bringup
ros2 launch robot_bringup robot.launch.py

# Check joint states
ros2 topic echo /joint_states --field position

# Test camera
ros2 run image_view image_view --ros-args --remap image:=/camera/rgb/image_raw
```

### 4. Safety System Test
```bash
# Test emergency stop
ros2 topic pub /emergency_stop std_msgs/Bool "data: true"

# Test safety limits
ros2 param list | grep safety
```

## Development Environment

### 1. IDE Setup
```bash
# Install VS Code for ARM64
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=arm64 signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code

# Install ROS extensions
code --install-extension ms-iot.vscode-ros
code --install-extension ms-python.python
```

### 2. Workspace Configuration
```bash
# Create development workspace
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws

# Clone essential packages
git clone https://github.com/physical-ai/humanoid-control.git src/humanoid_control
git clone https://github.com/physical-ai/humanoid-perception.git src/humanoid_perception
git clone https://github.com/physical-ai/humanoid-navigation.git src/humanoid_navigation

# Build workspace
colcon build --symlink-install --parallel-workers 4
source install/setup.bash
```

## Troubleshooting

### Common Issues

#### 1. Robot Not Responding
```bash
# Check if robot drivers are running
ps aux | grep robot

# Check ROS 2 nodes
ros2 node list

# Check hardware connections
dmesg | grep -i error
```

#### 2. Camera Not Working
```bash
# Check camera status
v4l2-ctl --list-devices

# Check if camera node is running
ros2 node list | grep camera

# Check camera topics
ros2 topic list | grep camera
```

#### 3. High CPU/GPU Usage
```bash
# Monitor system resources
htop
nvidia-smi

# Check for processes consuming excessive resources
ps aux --sort=-%cpu | head -20
```

## Maintenance and Updates

### 1. Regular Maintenance
```bash
# Update system packages regularly
sudo apt update && sudo apt upgrade -y

# Clean up unused packages
sudo apt autoremove && sudo apt autoclean

# Check disk usage
df -h
```

### 2. Backup Configuration
```bash
# Create backup of important configurations
tar -czf robot-config-backup-$(date +%Y%m%d).tar.gz \
  ~/.bashrc \
  ~/robot_ws/install \
  /etc/network/interfaces \
  ~/.ros/
```

## Learning Objectives

After completing this setup, you should be able to:
- Configure and operate the Physical AI Edge Kit
- Install and configure ROS 2 for humanoid robotics
- Calibrate robot sensors and kinematics
- Implement safety measures for robot operation
- Optimize system performance for real-time robotics
- Troubleshoot common hardware and software issues

## Next Steps

1. Complete robot calibration procedures
2. Run basic movement and perception tests
3. Begin the Physical AI & Humanoid Robotics curriculum
4. Develop your first robot application

## Resources

- [Physical AI Edge Kit Documentation](https://docs.physical-ai.com/edge-kit)
- [NVIDIA Jetson Documentation](https://docs.nvidia.com/jetson/)
- [ROS 2 Humble Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [Safety Guidelines for Humanoid Robotics](https://www.ieee.org/standards/safety-robotics.html)