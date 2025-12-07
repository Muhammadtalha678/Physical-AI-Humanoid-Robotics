---
sidebar_position: 6
title: "Digital Twin Workstation Setup"
---

# Digital Twin Workstation Setup

This guide provides instructions for setting up a Digital Twin workstation for Physical AI & Humanoid Robotics development. A Digital Twin workstation enables high-fidelity simulation and development of humanoid robotics applications.

## System Requirements

### Minimum Requirements
- **CPU**: Intel i7-10700K or AMD Ryzen 7 5800X
- **GPU**: NVIDIA RTX 3080 (10GB+ VRAM) or RTX 4080
- **RAM**: 32GB DDR4-3200MHz
- **Storage**: 1TB NVMe SSD
- **OS**: Ubuntu 22.04 LTS or Windows 11 Pro

### Recommended Requirements
- **CPU**: Intel i9-12900K or AMD Ryzen 9 5900X
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or RTX 6000 Ada
- **RAM**: 64GB DDR4-3600MHz or DDR5-4800MHz
- **Storage**: 2TB+ NVMe SSD (1TB system, 1TB+ for assets)
- **OS**: Ubuntu 22.04 LTS (preferred for robotics development)

### Network Requirements
- **Internet**: Stable connection with 50+ Mbps download
- **LAN**: Gigabit Ethernet recommended for robot communication
- **Bandwidth**: 100Mbps+ for simulation streaming (if applicable)

## Software Installation

### Ubuntu 22.04 Setup

#### 1. System Updates
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential cmake pkg-config
```

#### 2. NVIDIA GPU Drivers
```bash
# Check GPU compatibility
ubuntu-drivers devices

# Install recommended NVIDIA drivers
sudo ubuntu-drivers autoinstall

# Reboot after installation
sudo reboot
```

#### 3. ROS 2 Humble Hawksbill
```bash
# Set locale
locale  # check for UTF-8
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS 2 GPG key and repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt upgrade -y

# Install ROS 2 development tools
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

#### 4. NVIDIA Isaac Sim
```bash
# Install Isaac Sim prerequisites
sudo apt install python3.8-dev python3-pip
pip3 install --user setuptools==58.2.0

# Download Isaac Sim from NVIDIA Developer website
# Visit https://developer.nvidia.com/isaac-sim and download the latest version

# Extract and install (example for version 2023.1.1)
tar -xzf isaac-sim-2023.1.1.tar.gz
cd isaac-sim-2023.1.1
bash install.sh

# Add to environment
echo "export ISAACSIM_PATH=/path/to/isaac-sim-2023.1.1" >> ~/.bashrc
source ~/.bashrc
```

#### 5. Gazebo Garden
```bash
# Add Gazebo repository
sudo wget https://packages.gazebo.org/gazebo.gpg -O /usr/share/keyrings/gazebo-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.gazebo.org/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo.list > /dev/null

sudo apt update
sudo apt install gz-harmonic
```

#### 6. Development Tools
```bash
# Install Python development tools
pip3 install --user numpy scipy matplotlib
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install --user transformers openai

# Install development tools
sudo apt install git git-lfs vim nano htop
sudo apt install python3-dev python3-pip python3-venv
```

### Windows 11 Setup

#### 1. WSL2 with Ubuntu 22.04
```powershell
# Enable WSL and install Ubuntu 22.04
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Download and install Ubuntu 22.04 from Microsoft Store
# Set WSL2 as default version
wsl --set-default-version 2
```

#### 2. NVIDIA GPU Support in WSL2
```powershell
# Download and install NVIDIA WSL GPU driver
# Visit https://developer.nvidia.com/cuda/wsl and download the latest driver
# Install the downloaded .exe file
```

#### 3. Install ROS 2 and other tools in WSL2 Ubuntu
```bash
# Launch Ubuntu WSL2 and follow Ubuntu 22.04 setup instructions above
```

## Environment Configuration

### Environment Variables Setup
```bash
# Add to ~/.bashrc
echo "export ROS_DOMAIN_ID=1" >> ~/.bashrc
echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:~/models" >> ~/.bashrc
echo "export GAZEBO_RESOURCE_PATH=\$GAZEBO_RESOURCE_PATH:~/gazebo_resources" >> ~/.bashrc
echo "export ISAACSIM_PYTHON_PATH=/path/to/isaac-sim/python.sh" >> ~/.bashrc

source ~/.bashrc
```

### Docker Configuration (Optional but Recommended)
```bash
# Install Docker
sudo apt install docker.io
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-v2

# Reboot to apply changes
sudo reboot
```

## Development Environment Setup

### VS Code with Robotics Extensions
```bash
# Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code

# Install recommended extensions
code --install-extension ms-iot.vscode-ros
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension charliermarsh.ruff
```

### Workspace Setup
```bash
# Create ROS 2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build workspace
colcon build --symlink-install

# Source workspace
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Testing the Setup

### 1. Test ROS 2 Installation
```bash
# Open new terminal and test
ros2 topic list
ros2 run demo_nodes_cpp talker
# In another terminal: ros2 run demo_nodes_cpp listener
```

### 2. Test Gazebo Installation
```bash
gz sim --verbose
# Should launch Gazebo GUI
```

### 3. Test Isaac Sim (Linux)
```bash
# Launch Isaac Sim
./isaac-sim/python.sh -c "from omni.isaac.kit import SimulationApp; app = SimulationApp(); app.close()"
# Should initialize and close Isaac Sim without errors
```

## Performance Optimization

### GPU Configuration
```bash
# Check GPU status
nvidia-smi

# For Isaac Sim optimization, create a config file
mkdir -p ~/.config/isaac-sim
echo "exts: { omni.isaac.core: { enable_viewport_render: true, enable_scene_cache: false } }" > ~/.config/isaac-sim/config.yaml
```

### System Optimization
```bash
# Disable unnecessary services for better performance
sudo systemctl disable bluetooth
sudo systemctl disable ModemManager

# Configure swap for large simulations
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check if NVIDIA drivers are properly installed
nvidia-smi
# If not showing GPU, reinstall drivers:
sudo apt purge nvidia-* --autoremove
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### 2. ROS 2 Commands Not Found
```bash
# Check if ROS 2 is properly sourced
source /opt/ros/humble/setup.bash
echo $ROS_DISTRO  # Should show 'humble'
```

#### 3. Isaac Sim Rendering Issues
```bash
# Check OpenGL support
glxinfo | grep -i opengl
# Install additional packages if needed:
sudo apt install mesa-utils
```

## Learning Objectives

After completing this setup, you should be able to:
- Configure a high-performance workstation for humanoid robotics simulation
- Install and configure ROS 2, Gazebo, and Isaac Sim
- Set up development tools for robotics programming
- Verify the installation with basic tests
- Optimize the system for simulation performance

## Next Steps

1. Complete the basic ROS 2 tutorials
2. Explore Gazebo simulation environments
3. Try basic Isaac Sim examples
4. Begin the Physical AI & Humanoid Robotics curriculum

## Resources

- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/what_is_it.html)
- [Gazebo Garden Documentation](https://gazebosim.org/docs/harmonic)
- [Ubuntu Robotics Setup Guide](https://roboticsbackend.com/install-ros2-humble-ubuntu-22-04/)