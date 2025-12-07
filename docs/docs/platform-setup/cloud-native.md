---
sidebar_position: 8
title: "Cloud-Native Environment Setup"
---

# Cloud-Native Environment Setup

This guide provides instructions for setting up a cloud-native environment for Physical AI & Humanoid Robotics development. Cloud-native platforms provide scalable, distributed computing resources for simulation, training, and development of humanoid robotics applications.

## Cloud Platform Selection

### Recommended Platforms
- **AWS RoboMaker**: Managed service for robotics applications
- **Google Cloud Platform**: AI/ML services and Kubernetes
- **Microsoft Azure**: Azure IoT and machine learning services
- **NVIDIA CloudXR**: Cloud-based rendering for simulation
- **Self-hosted Kubernetes**: Custom cluster with GPU nodes

### Resource Requirements
- **GPU Instances**: NVIDIA T4, V100, A100, or H100 for simulation and training
- **Storage**: High-performance SSD storage for models and datasets
- **Network**: Low-latency, high-bandwidth connections
- **Memory**: 32GB+ RAM for simulation environments
- **CPU**: Multi-core processors for parallel processing

## Infrastructure Setup

### 1. Cloud Account Configuration
```bash
# AWS CLI setup
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# Enter Access Key ID, Secret Access Key, region (e.g., us-west-2)
```

### 2. Kubernetes Cluster Setup (Recommended)
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install cluster management tools
# For AWS EKS
curl -o kubectl-eks https://amazon-eks.s3.us-west-2.amazonaws.com/1.21.2/2021-07-05/bin/linux/amd64/kubectl
chmod +x ./kubectl-eks
sudo mv ./kubectl-eks /usr/local/bin/kubectl-eks

# For Google GKE
gcloud components install kubectl

# For Azure AKS
az aks install-cli
```

### 3. GPU Node Configuration
```bash
# Install NVIDIA Container Toolkit for GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Containerized Development Environment

### 1. Docker Configuration
```dockerfile
# Dockerfile for robotics development environment
FROM osrf/ros:humble-desktop-full-foxy

# Install NVIDIA runtime
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN echo 'PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' > /etc/environment

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install numpy scipy matplotlib \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    transformers openai \
    opencv-python open3d

# Install Gazebo and Isaac Sim dependencies
RUN apt-get update && apt-get install -y \
    gz-harmonic \
    libgazebo11-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS 2 workspace
RUN mkdir -p /opt/ros_ws/src
WORKDIR /opt/ros_ws

# Source ROS 2
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["bash"]
```

### 2. Docker Compose for Multi-Service Setup
```yaml
# docker-compose.yml
version: '3.8'
services:
  ros-core:
    build: .
    container_name: ros_core
    ports:
      - "11311:11311"
    environment:
      - ROS_DOMAIN_ID=1
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./workspace:/opt/ros_ws/src
    devices:
      - /dev/dri:/dev/dri
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  gazebo:
    build: .
    container_name: gazebo_sim
    ports:
      - "8080:8080"
    environment:
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./models:/root/.gazebo/models
    devices:
      - /dev/dri:/dev/dri
    depends_on:
      - ros-core
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  perception:
    build: .
    container_name: perception_node
    environment:
      - PYTHONPATH=/opt/ros_ws/install/lib/python3.10/site-packages:$PYTHONPATH
    volumes:
      - ./workspace:/opt/ros_ws/src
    depends_on:
      - ros-core
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Cloud-Specific Configurations

### AWS RoboMaker Setup
```bash
# Install AWS RoboMaker CLI
pip3 install awscli-plugin-endpoint
aws configure set plugin_endpoints.roborace https://roborace.us-west-2.amazonaws.com

# Create RoboMaker development environment
aws robomaker create-development-environment \
    --name MyRobotDevEnv \
    --repository-configuration "ecrConfiguration={imageTag=latest},managedECRRepository=MyRobotRepo" \
    --vpc-config "subnets=[subnet-12345678],securityGroupIds=[sg-12345678]" \
    --instance-type "General1Large"
```

### Google Cloud Platform Setup
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Create GKE cluster with GPU nodes
gcloud container clusters create robot-cluster \
    --zone us-central1-a \
    --node-pool default-pool \
    --num-nodes 1 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --enable-autoscaling \
    --min-nodes 1 \
    --max-nodes 3

# Get cluster credentials
gcloud container clusters get-credentials robot-cluster --zone us-central1-a
```

### Azure Setup
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Create AKS cluster with GPU nodes
az aks create \
    --resource-group robot-rg \
    --name robot-cluster \
    --node-count 1 \
    --node-vm-size Standard_NC6s_v3 \
    --enable-addons monitoring \
    --generate-ssh-keys

# Get cluster credentials
az aks get-credentials --resource-group robot-rg --name robot-cluster
```

## Kubernetes Configuration for Robotics

### 1. GPU Operator Installation
```bash
# Install NVIDIA GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
    --namespace gpu-operator --create-namespace \
    --set operator.defaultRuntime=containerd
```

### 2. Robotics Workload Configuration
```yaml
# robot-workload.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: robot-simulation
  labels:
    app: robot-sim
spec:
  replicas: 1
  selector:
    matchLabels:
      app: robot-sim
  template:
    metadata:
      labels:
        app: robot-sim
    spec:
      containers:
      - name: gazebo-sim
        image: robot-sim:latest
        ports:
        - containerPort: 11345
        env:
        - name: ROS_DOMAIN_ID
          value: "1"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "all"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: simulation-data
          mountPath: /simulation_data
      volumes:
      - name: simulation-data
        persistentVolumeClaim:
          claimName: robot-sim-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: robot-simulation-service
spec:
  selector:
    app: robot-sim
  ports:
    - protocol: TCP
      port: 11345
      targetPort: 11345
  type: LoadBalancer
```

### 3. Persistent Storage Setup
```yaml
# robot-storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: robot-sim-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

## Simulation and Training Environment

### 1. Isaac Sim in Cloud
```bash
# Create Isaac Sim deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isaac-sim
spec:
  replicas: 1
  selector:
    matchLabels:
      app: isaac-sim
  template:
    metadata:
      labels:
        app: isaac-sim
    spec:
      containers:
      - name: isaac-sim-container
        image: nvcr.io/nvidia/isaac-sim:latest
        ports:
        - containerPort: 55555
        - containerPort: 55556
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "all"
        - name: DISPLAY
          value: "headless"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: isaac-data
          mountPath: /isaac-sim-data
      volumes:
      - name: isaac-data
        persistentVolumeClaim:
          claimName: isaac-sim-pvc
```

### 2. Training Environment Setup
```yaml
# training-environment.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: robot-training-job
spec:
  template:
    spec:
      containers:
      - name: training-container
        image: robot-training:latest
        command: ["python3", "train_agent.py"]
        env:
        - name: TRAINING_ENV
          value: "cloud"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        volumeMounts:
        - name: training-data
          mountPath: /training_data
        - name: results
          mountPath: /results
      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: results
        persistentVolumeClaim:
          claimName: results-pvc
      restartPolicy: Never
```

## Remote Development Setup

### 1. VS Code Remote Development
```json
// .vscode/devcontainer.json
{
  "name": "Robotics Development",
  "dockerFile": "Dockerfile",
  "context": "..",
  "runArgs": [
    "--gpus", "all",
    "--env", "DISPLAY=${localEnv:DISPLAY}",
    "--volume", "/tmp/.X11-unix:/tmp/.X11-unix:rw"
  ],
  "workspaceFolder": "/opt/ros_ws",
  "workspaceMount": "source=${localWorkspaceFolder},target=/opt/ros_ws/src,type=bind",
  "settings": {
    "terminal.integrated.shell.linux": "/bin/bash"
  },
  "extensions": [
    "ms-iot.vscode-ros",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff"
  ],
  "forwardPorts": [11311],
  "postCreateCommand": "source /opt/ros/humble/setup.bash && colcon build --symlink-install"
}
```

### 2. SSH Configuration for Cloud Access
```bash
# SSH config for cloud instances
Host cloud-robot-dev
    HostName <your-instance-ip>
    User ubuntu
    IdentityFile ~/.ssh/robot-key.pem
    LocalForward 11311 localhost:11311
    LocalForward 8080 localhost:8080
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

## Security Configuration

### 1. Network Security
```bash
# Create security groups for robotics applications
# AWS example:
aws ec2 create-security-group \
    --group-name robot-sg \
    --description "Security group for robotics applications" \
    --vpc-id vpc-12345678

# Add rules for ROS 2 communication
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 11311 \
    --source-group sg-12345678

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol udp \
    --port 7400-7500 \
    --source-group sg-12345678
```

### 2. IAM Roles and Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::robot-data-bucket",
        "arn:aws:s3:::robot-data-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "robomaker:CreateSimulationApplication",
        "robomaker:CreateSimulationJob",
        "robomaker:DescribeSimulationJob"
      ],
      "Resource": "*"
    }
  ]
}
```

## Monitoring and Logging

### 1. Kubernetes Monitoring Setup
```yaml
# monitoring-stack.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus-deployment
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: monitoring
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: LoadBalancer
```

### 2. Application Logging
```yaml
# logging-stack.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluentd-deployment
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch-service.monitoring.svc.cluster.local"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
```

## Cost Optimization

### 1. Spot Instance Configuration
```bash
# AWS Spot Instance request for cost optimization
aws ec2 request-spot-instances \
    --spot-price "0.50" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification \
    InstanceType=m5.2xlarge,ImageId=ami-12345678,KeyName=my-key-pair
```

### 2. Auto-scaling Configuration
```yaml
# autoscaling.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: robot-simulation-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: robot-simulation
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Testing the Setup

### 1. Cluster Health Check
```bash
# Check cluster status
kubectl get nodes
kubectl get pods --all-namespaces

# Check GPU availability
kubectl get nodes -o json | jq '.items[].status.allocatable'

# Test GPU functionality
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
kubectl run gpu-test --rm -t -i --restart=Never --image=nvidia/cuda:11.0.3-base-ubuntu20.04 --limits=nvidia.com/gpu=1 -- nvidia-smi
```

### 2. ROS 2 Communication Test
```bash
# Deploy ROS 2 test containers
kubectl apply -f ros-test-deployment.yaml

# Test ROS 2 communication
kubectl exec -it ros-core-pod -- ros2 topic list
kubectl exec -it ros-core-pod -- ros2 run demo_nodes_cpp talker &
kubectl exec -it ros-core-pod -- ros2 run demo_nodes_cpp listener
```

### 3. Simulation Environment Test
```bash
# Deploy simulation environment
kubectl apply -f simulation-environment.yaml

# Test Gazebo simulation
kubectl port-forward gazebo-pod 8080:8080
# Access http://localhost:8080 to view simulation
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Available in Pod
```bash
# Check if GPU operator is running
kubectl get pods -n gpu-operator

# Verify GPU resources are available
kubectl describe nodes | grep nvidia.com/gpu

# Check device plugin status
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
```

#### 2. ROS 2 Communication Issues
```bash
# Check network policies
kubectl get networkpolicies

# Verify ROS 2 domain configuration
kubectl exec -it pod-name -- printenv | grep ROS

# Test network connectivity between pods
kubectl exec -it pod1 -- ping pod2-ip
```

#### 3. Performance Issues
```bash
# Monitor resource usage
kubectl top nodes
kubectl top pods

# Check for resource limits
kubectl describe pod pod-name

# Scale resources if needed
kubectl patch deployment deployment-name -p '{"spec":{"replicas":2}}'
```

## Learning Objectives

After completing this setup, you should be able to:
- Configure cloud-native infrastructure for robotics applications
- Deploy containerized robotics environments in Kubernetes
- Set up GPU-accelerated simulation and training environments
- Implement security and monitoring for cloud robotics
- Optimize costs and performance in cloud environments
- Troubleshoot cloud-native robotics deployments

## Next Steps

1. Deploy your first robotics application in the cloud
2. Set up CI/CD pipelines for robotics development
3. Implement distributed training for robot learning
4. Begin the Physical AI & Humanoid Robotics curriculum in the cloud environment

## Resources

- [Kubernetes for Robotics](https://github.com/AutonomyLab/robotics_deployment_kubernetes)
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [AWS RoboMaker Developer Guide](https://docs.aws.amazon.com/robomaker/latest/dg/what-is-robomaker.html)
- [Google Cloud for Robotics](https://cloud.google.com/solutions/iot/robotics)
- [Azure IoT Robotics Solutions](https://azure.microsoft.com/en-us/solutions/iot/robotics/)